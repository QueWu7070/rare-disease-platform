'use strict';

// Caliper workload module for the data-sharing chaincode.
// In 'create' mode it submits CreateAsset transactions with unique ids;
// in 'read' mode it evaluates ReadAsset against previously created ids.
// Target: Hyperledger Caliper v0.4.2.

const { WorkloadModuleBase } = require('@hyperledger/caliper-core');

class DataSharingWorkload extends WorkloadModuleBase {
    constructor() {
        super();
        this.mode = 'create';
        this.txIndex = 0;
    }

    async initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext) {
        await super.initializeWorkloadModule(
            workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext,
        );
        this.mode = roundArguments.mode || 'create';
    }

    async submitTransaction() {
        this.txIndex++;
        // Namespace ids by worker to avoid key collisions across workers.
        const assetId = `asset_${this.workerIndex}_${this.txIndex}`;

        if (this.mode === 'create') {
            const request = {
                contractId: 'data_sharing_cc',
                contractFunction: 'CreateAsset',
                contractArguments: [
                    assetId,
                    `0x${this.txIndex.toString(16).padStart(64, '0')}`,
                    `benchmark asset ${assetId}`,
                ],
                invokerIdentity: 'User1',
                readOnly: false,
            };
            await this.sutAdapter.sendRequests(request);
        } else {
            const request = {
                contractId: 'data_sharing_cc',
                contractFunction: 'ReadAsset',
                contractArguments: [`asset_${this.workerIndex}_${this.txIndex}`],
                invokerIdentity: 'User1',
                readOnly: true,
            };
            await this.sutAdapter.sendRequests(request);
        }
    }

    async cleanupWorkloadModule() {
        // No cleanup required; ledger state is discarded when the network is torn down.
    }
}

function createWorkloadModule() {
    return new DataSharingWorkload();
}

module.exports.createWorkloadModule = createWorkloadModule;
