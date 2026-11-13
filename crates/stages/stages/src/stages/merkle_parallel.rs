//! Parallel full-rebuild of the state trie for the Merkle stage.
//!
//! The upstream rebuild walks every account and computes each storage root on one core. Storage
//! roots are independent of each other, so they are computed across all cores with rayon in
//! account-ordered chunks, then feed the account leaves into a single [`HashBuilder`] in order.
//!
//! All workers share the stage's own RW transaction: reth opens MDBX with `MDBX_NOSTICKYTHREADS`
//! and every cursor op goes through the transaction lock, so DB reads are serialized while the
//! keccak and RLP work runs in parallel.

use alloy_primitives::B256;
use alloy_rlp::Encodable;
use rayon::prelude::*;
use reth_db_api::{transaction::DbTx, DatabaseError};
use reth_provider::{ProviderError, ProviderResult};
use reth_trie::{
    hashed_cursor::{HashedCursor, HashedCursorFactory},
    metrics::TrieRootMetrics,
    trie_cursor::noop::NoopTrieCursorFactory,
    updates::TrieUpdates,
    HashBuilder, Nibbles, StorageRoot, TrieType, EMPTY_ROOT_HASH,
};
use reth_trie_db::DatabaseHashedCursorFactory;
use std::time::Instant;
use tracing::{debug, info};

/// Accounts per parallel batch. Bounds memory to one batch of storage trie updates.
pub const PARALLEL_MERKLE_CHUNK: usize = 16_384;

/// Rebuilds the state trie from the hashed tables, computing storage roots on all cores.
///
/// Assumes `AccountsTrie`/`StoragesTrie` were cleared: no stored trie nodes are read. Calls
/// `write` with each batch of storage trie updates, then once with the account trie nodes.
/// Returns the state root and the number of hashed entries walked.
//
// This runs to completion in one stage execution with no intermediate checkpoints, so a crash
// during the rebuild starts it over.
pub fn parallel_rebuild_state_root<TX: DbTx + Sync>(
    tx: &TX,
    mut write: impl FnMut(TrieUpdates) -> ProviderResult<()>,
) -> ProviderResult<(B256, u64)> {
    let started = Instant::now();
    let factory = DatabaseHashedCursorFactory::new(tx);
    let metrics = TrieRootMetrics::new(TrieType::Storage);
    let threads = rayon::current_num_threads();
    info!(target: "sync::stages::merkle::exec", threads, "Parallel merkle rebuild started");

    let mut account_cursor = factory.hashed_account_cursor()?;
    let mut next = account_cursor.seek(B256::ZERO)?;
    let mut hash_builder = HashBuilder::default().with_updates(true);
    let mut leaf = Vec::new();
    let (mut accounts, mut slots) = (0u64, 0u64);

    while next.is_some() {
        let mut chunk = Vec::with_capacity(PARALLEL_MERKLE_CHUNK);
        while let Some(entry) = next {
            chunk.push(entry);
            next = account_cursor.next()?;
            if chunk.len() == PARALLEL_MERKLE_CHUNK {
                break
            }
        }

        // Storage roots for the whole chunk, in parallel. `par_iter().map().collect()` keeps
        // input order, which the hash builder needs.
        let roots = chunk
            .par_iter()
            .map(|(address, _)| {
                let (root, walked, updates) = StorageRoot::new_hashed(
                    NoopTrieCursorFactory::default(),
                    DatabaseHashedCursorFactory::new(tx),
                    *address,
                    Default::default(),
                    metrics.clone(),
                )
                .root_with_updates()
                .map_err(DatabaseError::from)?;
                Ok((root, walked, updates))
            })
            .collect::<Result<Vec<_>, ProviderError>>()?;

        let mut storage_updates = TrieUpdates::default();
        for ((address, account), (root, walked, updates)) in chunk.into_iter().zip(roots) {
            leaf.clear();
            account.into_trie_account(root).encode(&mut leaf);
            hash_builder.add_leaf(Nibbles::unpack(address), &leaf);
            if root != EMPTY_ROOT_HASH {
                storage_updates.insert_storage_updates(address, updates);
            }
            accounts += 1;
            slots += walked as u64;
        }
        write(storage_updates)?;

        debug!(
            target: "sync::stages::merkle::exec",
            accounts,
            slots,
            elapsed = ?started.elapsed(),
            "Parallel merkle progress"
        );
    }

    let root = hash_builder.root();
    let mut account_updates = TrieUpdates::default();
    account_updates.finalize(hash_builder, Default::default(), Default::default());
    write(account_updates)?;

    info!(
        target: "sync::stages::merkle::exec",
        %root,
        accounts,
        slots,
        threads,
        elapsed = ?started.elapsed(),
        "Parallel merkle rebuild finished"
    );
    Ok((root, accounts + slots))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::U256;
    use rand::Rng;
    use reth_db_api::{tables, transaction::DbTxMut};
    use reth_primitives_traits::{Account, StorageEntry};
    use reth_provider::test_utils::create_test_provider_factory;
    use reth_testing_utils::generators;
    use reth_trie::StateRoot;

    #[test]
    fn parallel_rebuild_matches_sequential() {
        let factory = create_test_provider_factory();
        let provider = factory.provider_rw().unwrap();
        let tx = provider.tx_ref();
        let mut rng = generators::rng();

        // More than one chunk, a mix of EOAs and contracts with varying storage sizes.
        for i in 0..PARALLEL_MERKLE_CHUNK as u64 + 1_000 {
            let address = B256::random_with(&mut rng);
            let account =
                Account { nonce: i, balance: U256::from(rng.random::<u64>()), bytecode_hash: None };
            tx.put::<tables::HashedAccounts>(address, account).unwrap();
            if i % 3 == 0 {
                for _ in 0..rng.random_range(1..40) {
                    let entry = StorageEntry {
                        key: B256::random_with(&mut rng),
                        value: U256::from(rng.random::<u64>() | 1),
                    };
                    tx.put::<tables::HashedStorages>(address, entry).unwrap();
                }
            }
        }

        let (expected_root, expected_updates) =
            StateRoot::new(NoopTrieCursorFactory::default(), DatabaseHashedCursorFactory::new(tx))
                .root_with_updates()
                .unwrap();

        let mut updates = TrieUpdates::default();
        let (root, _) = parallel_rebuild_state_root(tx, |batch| {
            updates.extend(batch);
            Ok(())
        })
        .unwrap();

        assert_eq!(root, expected_root);
        assert_eq!(updates.into_sorted(), expected_updates.into_sorted());
    }
}
