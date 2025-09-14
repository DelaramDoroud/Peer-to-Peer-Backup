# Peer-to-Peer Backup Simulation

This project is part of the **Distributed Computing** course and simulates a **Peer-to-Peer (P2P) backup system** using **erasure coding**, comparing it against a **Client-Server model**, and explores extensions to improve resilience and fairness.

---

## 📌 Project Overview
The main objective is to analyze **long-term data preservation** in distributed systems where nodes frequently join, leave, or fail.  
Key aspects include:
- **Erasure Coding**: Data is divided into *n* blocks and can be recovered with *k* blocks.  
- **Node Dynamics**: Peers have variable uptime, downtime, lifetime, and storage limits.  
- **System Evaluation**: Performance is assessed through metrics like *data loss over time*, *local blocks*, and *backed-up blocks*.  

---

## 📂 Project Structure
- `p2p_backup.py` → Core P2P backup implementation  
- `client_server_backup.py` → Client-Server comparison  
- `priority_extension.py` → Priority-based node selection  
- `selfish_node_extension.py` → Selfish node behavior & Tit-for-Tat strategy  
- `plots/` → Simulation results and comparison graphs  
