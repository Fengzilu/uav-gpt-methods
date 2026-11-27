# UAV-GPT

Source code accompanying *UAV-GPT: A General Foundation Model Framework for UAV RF Monitoring*.

| File | Purpose |
| --- | --- |
| [scripts/preprocess_drone_v51.py](scripts/preprocess_drone_v51.py) | Read the four IQ sources, select and normalize windows, and write their metadata and shards. |
| [scripts3/preprocess_main.py](scripts3/preprocess_main.py) | Prepare the four-source pretraining input using a recording-level index and caller-supplied paths. |
| [scripts3/adaptive_branch_lengths.py](scripts3/adaptive_branch_lengths.py) | Learn the three input lengths from unlabeled waveform fidelity. |
| [scripts3/adaptive_pretrain_routing.py](scripts3/adaptive_pretrain_routing.py) | Fit and calibrate the router on pretraining recordings and prepare an adaptive input index. |
| [scripts3/pretrain_engine.py](scripts3/pretrain_engine.py) | Train the rate-specific front ends and shared encoder with the staged curriculum and self-supervised objectives. |

Preprocessing takes the raw-data paths and the recording index as inputs. Pretraining takes a preprocessed IQ shard directory and can use an explicit adaptive index.

## Data sources

- [DroneRFa](https://doi.org/10.11999/JEIT230570) provides recorded UAV radio-frequency signals.
- [RF Signals of UAVs](https://www.kaggle.com/datasets/xcz74741/rf-signals-of-uavs) provides UAV signal recordings.
- [Gent Technology Recognition](https://github.com/JaronFontaine/Technology-Recognition-dataset-of-real-life-LTE-WiFi-and-DVB-T) provides recorded LTE, Wi-Fi, and DVB-T IQ. [IEEE DataPort](https://doi.org/10.21227/5G7M-GF53)
- [RadioML 2018.01A](https://www.deepsig.ai/datasets/) provides labeled IQ frames for digital and analog modulations.
