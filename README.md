# SIH26125 — Blockchain-Based Secure Platform for Identity, Access Control & Digital Asset Management

A decentralized platform that unifies **Self-Sovereign Identity (DID)**, **NFT-based digital asset ownership**, and **on-chain Role-Based Access Control (RBAC)** into a single trustless, tamper-proof system — built for organizations that need verifiable identity, controlled asset issuance, and a fully auditable trail of every action.

---

## Problem Statement

**SIH26125** — Traditional identity and access management systems are centralized, making them vulnerable to breaches, identity theft, and single points of failure. Asset ownership records are similarly fragmented, making authenticity and provenance hard to verify.

This project solves that by combining:
- **Decentralized Identifiers (DIDs)** for self-sovereign, cryptographically verifiable identity
- **NFTs (ERC-721 / ERC-1155)** for unique, traceable, immutable digital asset ownership
- **On-chain RBAC** to enforce who can create, allocate, and manage assets
- **Immutable event logs** for full transparency and auditability

---

## Key Features

| Feature | Description |
|---|---|
| 🔑 Self-Sovereign Identity | Users authenticate via wallet signatures (`did:ethr:0x...`) — no central password database |
| 🖼️ NFT Asset Management | Assets minted as ERC-721/1155 tokens, hard-linked to a user's DID |
| 🛡️ Restricted Minting | Only `Admin` / `Manager` roles can mint and allocate NFTs — no public minting |
| 🔒 Soulbound Option | Assets can be made non-transferable to guarantee immutable ownership |
| 👥 On-Chain RBAC | Roles: `Admin`, `Manager`, `Auditor`, `User` — enforced via OpenZeppelin `AccessControl.sol` |
| 📜 Immutable Audit Trail | Every identity, minting, allocation, transfer, and role change emits an on-chain event |
| 🔍 Auditor Dashboard | Read-only role to trace full provenance and access history of any asset |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (dApp)                       │
│   Wallet Connect · Identity Onboarding · Asset Dashboard      │
│              · Admin Panel · Auditor View                     │
└───────────────────────────┬───────────────────────────────────┘
                            │ Web3 / Ethers.js
┌───────────────────────────┴───────────────────────────────────┐
│                     Smart Contract Layer                      │
│                                                                 │
│  ┌────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │  DID Registry   │  │  RBAC Controller  │  │  NFT Asset      │ │
│  │  (Identity Mgmt)│  │  (AccessControl)   │  │  Contract       │ │
│  └────────────────┘  └──────────────────┘  └────────────────┘ │
│                                                                 │
│                  Events → Immutable Audit Log                  │
└───────────────────────────┬───────────────────────────────────┘
                            │
                  ┌─────────┴─────────┐
                  │   Blockchain       │
                  │ (Ethereum / L2 /   │
                  │  Polygon testnet)  │
                  └───────────────────┘
```

---

## Roles & Permissions

| Role | Privileges |
|---|---|
| **Admin** | Top-level authority. Grants/revokes roles (e.g., promotes a User to Manager). |
| **Manager** | Mints NFTs and allocates them directly to User DIDs. |
| **Auditor** | Read-only access. Queries provenance, ownership, and access history of any asset. |
| **User** | Owns a DID, views allocated assets, authenticates via cryptographic signature. |

---

## Tech Stack

- **Smart Contracts:** Solidity, OpenZeppelin (`AccessControl.sol`, `ERC721`/`ERC1155`)
- **Identity Standard:** W3C DID (`did:ethr` method)
- **Blockchain:** Ethereum-compatible (testnet: Sepolia / Polygon Mumbai)
- **Dev Framework:** Hardhat / Foundry
- **Frontend:** React + Ethers.js / Wagmi + MetaMask (or WalletConnect)
- **Indexing/Audit:** The Graph (or direct event log queries) for the auditor dashboard

---

## Core Smart Contract Events

```solidity
event IdentityCreated(address indexed userDID, uint256 timestamp);
event RoleGranted(bytes32 role, address account, address admin);
event RoleRevoked(bytes32 role, address account, address admin);
event AssetMinted(uint256 indexed nftID, address indexed minter, string metadataURI);
event AssetAllocated(uint256 indexed nftID, address indexed userDID, address manager);
event AssetTransferred(uint256 indexed nftID, address indexed from, address indexed to);
```

All events are indexed and permanently queryable, forming the tamper-proof audit trail required by the problem statement.

---

## Workflow

1. **Onboarding** — User connects wallet; a DID is generated and registered on-chain.
2. **Verification** — Admin verifies real-world credentials off-chain, then grants the `User` role on-chain.
3. **Asset Creation** — Manager inputs asset details and mints it as an NFT.
4. **Allocation** — The NFT is transferred to the User's DID; an `AssetAllocated` event is emitted.
5. **Auditing** — Auditor queries the dashboard to view the full immutable history of any asset or identity.

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/<your-org>/sih26125-blockchain-identity.git
cd sih26125-blockchain-identity

# Install dependencies
npm install

# Compile contracts
npx hardhat compile

# Run tests
npx hardhat test

# Deploy to testnet
npx hardhat run scripts/deploy.js --network sepolia

# Start frontend
cd frontend
npm install
npm run dev
```

---

## Project Structure

```
├── contracts/
│   ├── IdentityRegistry.sol      # DID registration & resolution
│   ├── RBACController.sol        # Role-based access control (AccessControl.sol)
│   ├── AssetNFT.sol              # ERC-721/1155 minting, allocation, soulbound logic
│   └── interfaces/
├── scripts/
│   └── deploy.js
├── test/
│   ├── identity.test.js
│   ├── rbac.test.js
│   └── nft.test.js
├── frontend/
│   ├── src/
│   │   ├── pages/                # Onboarding, Dashboard, Admin, Auditor views
│   │   └── hooks/                # useWallet, useContract, useEvents
│   └── package.json
└── README.md
```

---

## Security Considerations

- Uses OpenZeppelin's audited `AccessControl.sol` rather than custom RBAC logic
- Minting restricted via `onlyRole(MANAGER_ROLE)` modifiers
- Optional soulbound (non-transferable) mode to prevent unauthorized resale/theft of allocated assets
- All privileged actions require an on-chain role check before execution
- No private keys or credentials ever touch a central server — authentication is signature-based

---

## Future Scope

- Cross-chain DID resolution for multi-chain asset portability
- Zero-knowledge proofs for privacy-preserving identity verification
- Integration with government/enterprise KYC providers for off-chain verification
- Mobile wallet support for citizen-facing deployments

---

## Team

THE MORPHS

## License

MIT
