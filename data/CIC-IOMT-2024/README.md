## 📂 Dataset Details: CICIOMT2024

To ensure the authenticity and reproducibility of our experiments, we provide the detailed specification of the **CICIOMT2024** subset used in this study. The dataset focuses on IoT-specific attack vectors (MQTT, UDP, TCP) and benign traffic.

### 1. Feature Space
Our model utilizes a refined set of **flow-based features** extracted to capture network behavior patterns. The features are categorized as follows:

* **Flow Identifiers:**
    `IPV4_SRC_ADDR`, `L4_SRC_PORT`, `IPV4_DST_ADDR`, `L4_DST_PORT`, `PROTOCOL`, `L7_PROTO`
* **Volumetric & Time Statistics:**
    `IN_BYTES`, `IN_PKTS`, `OUT_BYTES`, `OUT_PKTS`, `FLOW_DURATION_MILLISECONDS`
* **TCP/IP & TTL Behavior:**
    `TCP_FLAGS`, `TCP_WIN_MAX_IN`, `TCP_WIN_MAX_OUT`, `MIN_TTL`, `MAX_TTL`
* **Packet Length Statistics:**
    `LONGEST_FLOW_PKT`, `SHORTEST_FLOW_PKT`, `MIN_IP_PKT_LEN`, `MAX_IP_PKT_LEN`
* **Packet Size Distribution (Histogram):**
    `NUM_PKTS_UP_TO_128_BYTES`, `NUM_PKTS_128_TO_256_BYTES`, `NUM_PKTS_256_TO_512_BYTES`, `NUM_PKTS_512_TO_1024_BYTES`, `NUM_PKTS_1024_TO_1514_BYTES`

### 2. Class Distribution Statistics
The dataset reflects a realistic, imbalanced network environment. The specific breakdown of the samples used in our training/testing set is detailed below:

| Label Name | Count | Ratio (%) | Description |
| :--- | :---: | :---: | :--- |
| **MQTT-DDoS-Publish_Flood_train** | 67,213 | 44.54% | IoT-specific DDoS targeting MQTT protocol |
| **TCP_IP-DDoS-UDP_train** | 26,083 | 17.28% | Volumetric UDP Flood (DDoS) |
| **TCP_IP-DoS-UDP_train** | 20,342 | 13.48% | Volumetric UDP Flood (DoS) |
| **MQTT-DoS-Connect_Flood_train** | 14,205 | 9.41% | IoT-specific DoS targeting MQTT connections |
| **Benign_train** | 10,884 | 7.21% | Normal background traffic |
| **Recon-VulScan_train** | 6,432 | 4.26% | Vulnerability Scanning (Reconnaissance) |
| **TCP_IP-DDoS-ICMP_train** | 5,762 | 3.82% | ICMP Flood (DDoS) |
| **Total Samples** | **150,921** | **100.0%** | - |

> **Note on Other Datasets:** Apart from the CICIOMT2024 specifications detailed above, all other datasets referenced in this work utilize their respective standard public feature sets and full data availability.
