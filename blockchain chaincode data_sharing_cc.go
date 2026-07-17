// Package main implements a Hyperledger Fabric chaincode for rare-disease data
// sharing. It records data-sharing assets on the ledger and enforces a simple
// access-control check based on the requesting organisation's MSP ID.
//
// Target runtime: Hyperledger Fabric v2.2, Go 1.16,
// fabric-contract-api-go v1.1.1, CouchDB v3.1 as the state database.
package main

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// SmartContract provides functions for managing data-sharing assets.
type SmartContract struct {
	contractapi.Contract
}

// DataAsset represents a shared rare-disease dataset record.
type DataAsset struct {
	ID          string `json:"id"`
	Owner       string `json:"owner"`       // owning organisation MSP ID
	DataHash    string `json:"dataHash"`    // hash of the off-chain payload
	Description string `json:"description"`
	AccessList  []string `json:"accessList"` // MSP IDs granted read access
	Timestamp   string `json:"timestamp"`
}

// InitLedger seeds the ledger with an initial set of assets.
func (s *SmartContract) InitLedger(ctx contractapi.TransactionContextInterface) error {
	assets := []DataAsset{
		{
			ID:          "asset1",
			Owner:       "Org1MSP",
			DataHash:    "0x0000000000000000000000000000000000000000000000000000000000000000",
			Description: "Initial rare-disease cohort record",
			AccessList:  []string{"Org1MSP"},
			Timestamp:   time.Now().UTC().Format(time.RFC3339),
		},
	}

	for _, asset := range assets {
		assetJSON, err := json.Marshal(asset)
		if err != nil {
			return err
		}
		if err := ctx.GetStub().PutState(asset.ID, assetJSON); err != nil {
			return fmt.Errorf("failed to put asset %s: %v", asset.ID, err)
		}
	}
	return nil
}

// CreateAsset issues a new data-sharing asset owned by the submitting org.
func (s *SmartContract) CreateAsset(
	ctx contractapi.TransactionContextInterface,
	id string,
	dataHash string,
	description string,
) error {
	exists, err := s.AssetExists(ctx, id)
	if err != nil {
		return err
	}
	if exists {
		return fmt.Errorf("asset %s already exists", id)
	}

	owner, err := ctx.GetClientIdentity().GetMSPID()
	if err != nil {
		return fmt.Errorf("failed to read client MSP ID: %v", err)
	}

	asset := DataAsset{
		ID:          id,
		Owner:       owner,
		DataHash:    dataHash,
		Description: description,
		AccessList:  []string{owner},
		Timestamp:   time.Now().UTC().Format(time.RFC3339),
	}
	assetJSON, err := json.Marshal(asset)
	if err != nil {
		return err
	}
	return ctx.GetStub().PutState(id, assetJSON)
}

// ReadAsset returns the asset if the caller is on its access list.
func (s *SmartContract) ReadAsset(
	ctx contractapi.TransactionContextInterface,
	id string,
) (*DataAsset, error) {
	assetJSON, err := ctx.GetStub().GetState(id)
	if err != nil {
		return nil, fmt.Errorf("failed to read asset %s: %v", id, err)
	}
	if assetJSON == nil {
		return nil, fmt.Errorf("asset %s does not exist", id)
	}

	var asset DataAsset
	if err := json.Unmarshal(assetJSON, &asset); err != nil {
		return nil, err
	}

	caller, err := ctx.GetClientIdentity().GetMSPID()
	if err != nil {
		return nil, fmt.Errorf("failed to read client MSP ID: %v", err)
	}
	if !contains(asset.AccessList, caller) {
		return nil, fmt.Errorf("access denied for %s on asset %s", caller, id)
	}
	return &asset, nil
}

// GrantAccess adds an organisation to an asset's access list. Only the owner
// may grant access.
func (s *SmartContract) GrantAccess(
	ctx contractapi.TransactionContextInterface,
	id string,
	grantee string,
) error {
	assetJSON, err := ctx.GetStub().GetState(id)
	if err != nil {
		return fmt.Errorf("failed to read asset %s: %v", id, err)
	}
	if assetJSON == nil {
		return fmt.Errorf("asset %s does not exist", id)
	}

	var asset DataAsset
	if err := json.Unmarshal(assetJSON, &asset); err != nil {
		return err
	}

	caller, err := ctx.GetClientIdentity().GetMSPID()
	if err != nil {
		return fmt.Errorf("failed to read client MSP ID: %v", err)
	}
	if caller != asset.Owner {
		return fmt.Errorf("only owner %s may grant access", asset.Owner)
	}

	if !contains(asset.AccessList, grantee) {
		asset.AccessList = append(asset.AccessList, grantee)
	}
	updated, err := json.Marshal(asset)
	if err != nil {
		return err
	}
	return ctx.GetStub().PutState(id, updated)
}

// AssetExists reports whether an asset with the given id is on the ledger.
func (s *SmartContract) AssetExists(
	ctx contractapi.TransactionContextInterface,
	id string,
) (bool, error) {
	assetJSON, err := ctx.GetStub().GetState(id)
	if err != nil {
		return false, fmt.Errorf("failed to read asset %s: %v", id, err)
	}
	return assetJSON != nil, nil
}

// GetAllAssets returns every asset on the ledger (range query).
func (s *SmartContract) GetAllAssets(
	ctx contractapi.TransactionContextInterface,
) ([]*DataAsset, error) {
	resultsIterator, err := ctx.GetStub().GetStateByRange("", "")
	if err != nil {
		return nil, err
	}
	defer resultsIterator.Close()

	var assets []*DataAsset
	for resultsIterator.HasNext() {
		queryResponse, err := resultsIterator.Next()
		if err != nil {
			return nil, err
		}
		var asset DataAsset
		if err := json.Unmarshal(queryResponse.Value, &asset); err != nil {
			return nil, err
		}
		assets = append(assets, &asset)
	}
	return assets, nil
}

func contains(list []string, target string) bool {
	for _, v := range list {
		if v == target {
			return true
		}
	}
	return false
}

func main() {
	chaincode, err := contractapi.NewChaincode(&SmartContract{})
	if err != nil {
		fmt.Printf("Error creating data-sharing chaincode: %v\n", err)
		return
	}
	if err := chaincode.Start(); err != nil {
		fmt.Printf("Error starting data-sharing chaincode: %v\n", err)
	}
}
