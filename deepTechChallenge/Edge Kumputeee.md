
## 1. System Architecture: Visual Plant Disease Detection

This system operates as a pure vision-based Edge AI device. It captures leaf structures locally, executes a quantized Convolutional Neural Network (CNN), and outputs diagnostic classifications entirely offline.

### Hardware Layer

- **Core Microcontroller:** ESP32-S3-WROOM-1 module (configured with 8MB Octal SPI PSRAM and 16MB Flash).
    
- **Vision Sensor:** OV2640 camera module connected via the native 8-bit parallel Digital Video Port (DVP) interface.
    
- **Power Subsystem:** Dedicated low-dropout (LDO) regulator with high transient response (e.g., AP2112K) to eliminate voltage drops and analog noise during camera DMA capture and CPU spikes.
    

### Software & Task Architecture (FreeRTOS)

- **Core 0 (Application Logic & I/O):** Manages Direct Memory Access (DMA) frame buffer accumulation, image downsampling, and status communications.
    
- **Core 1 (Isolated TinyML Inference):** Dedicated strictly to running the inference loop. This isolation provides deterministic execution timing, which is crucial for the paper’s performance metrics.
    
- **Storage Framework:** Local logs, inference metrics (latency, layer execution ticks), and exception benchmarks are written directly to a local **LittleFS** partition on the SPI Flash.
    

### TinyML Integration Pipeline

- **Inference Engine:** TensorFlow Lite for Microcontrollers (TFLM) compiled with Espressif’s **ESP-NN** assembly-optimized libraries. This allows the model to leverage the ESP32-S3's **Processor Instruction Extension (PIE)** vector instructions, accelerating 2D convolutions by up to $14\times$ compared to standard ANSI C implementations.
    
- **Model Target Profile:** A scaled-down MobileNetV2 architecture with a width multiplier of $\alpha = 0.25$ and an input shape scaled down quadratically to $96 \times 96 \times 3$ (RGB) or $128 \times 128 \times 1$ (Grayscale).
    
- **Memory Optimization:** Quantized to **INT8 precision**. The raw model weights reside in Flash, while the dynamic `Tensor Arena` (allocated at $\sim300\text{ KB}$) is placed in internal SRAM to ensure maximum bus performance, avoiding the speed penalty of external PSRAM where possible.
    

### Data Sourcing & Training Pipeline

1. **Sourcing:** Extracted from the public _PlantVillage_ dataset, focused strictly on a subset of 3–4 high-impact localized crop diseases to limit output layer complexity.
    
2. **Preprocessing & Augmentation:** Input pipelines in Python utilize aggressive resolution scaling, normalization, and artificial lighting/rotation shifts to simulate field environments.
    
3. **Quantization:** Post-Training Quantization (PTQ) via the `TFLiteConverter` using a representative calibration dataset to map activations to exact 8-bit integers without destroying model accuracy.
    
4. **Deployment Artifact:** Converted to a static C byte array using standard hex utilities (`xxd -i`) and integrated as a header file within the PlatformIO build structure.
    

## 2. System Architecture: Smart Self-Checkout System

By pivoting to a **pure time-series weight-based TinyML architecture**, we eliminate the overhead of a camera and shift the focus entirely to high-frequency edge signal processing and pattern recognition.

### 1. Hardware Layer

- **Core Microcontroller:** ESP32-S3-WROOM-1 (8MB PSRAM / 16MB Flash).
    
- **Sensor Interface:** **HX711 24-bit ADC** module. The physical `RATE` pin (Pin 15) is hardware-strapped to `VCC` to force an **80 Hz sampling rate**, providing the necessary temporal resolution for transient impact captures.
    
- **Transducer:** Four parallel shear-beam or strain-gauge load cells configured in a Wheatstone bridge layout beneath the cart/basket platform.
    
- **State Toggle Input:** A rugged, panel-mounted momentary push-button tied to a native ESP32-S3 GPIO (configured with an internal pull-up resistor and hardware/software debouncing) to serve as the **Disposal/Item Removal trigger**.
    

### 2. Software & Multi-Core Architecture (FreeRTOS)

To ensure zero data loss from the high-frequency ADC stream, processing tasks are strictly split across both Xtensa LX7 cores.

```
                  [Core 0: Ingestion & Filtering]
                                 │
                   (HX711 @ 80Hz Continuous Stream)
                                 │
                        [dF/dt Threshold?] 
                                 │
                     ┌───────────┴───────────┐
                    YES                      NO
                     │                       │
         [Capture 64-Sample Window]   (Maintain Baseline)
                     │
        ┌────────────┴────────────┐
   [Button Pressed?]       [Button Idle?]
        │                         │
  (STATE_REMOVE)            (STATE_ADD)
        │                         │
        └────────────┬────────────┘
                     │
          (Push Window via Queue)
                     │
                     ▼
       [Core 1: TinyML 1D-CNN Inference]
                     │
      (Output: Predicted Item Class)
                     │
                     ▼
       [Core 0: Inventory Manifest FSM]
```

##|# Core 0: Data Ingestion, Filtering, and State Control

- **High-Frequency Sampling:** Reads the HX711 via a bit-bashed or hardware-timed I/O loop at 80 Hz.
    
- **Digital Filtering:** Applies a fast rolling median filter (window size = 3) to strip out high-frequency stochastic electrical noise without dampening the structural features of the impact impulse.
    
- **Trigger Mechanism:** Continuously monitors the first derivative of the force profile:
    

$$\left| \frac{dF}{dt} \right| > \text{Threshold}_{\text{impact}}$$

- When this threshold is breached, Core 0 opens a fixed-size capture window of **64 samples** ($\sim800\text{ ms}$ of continuous data covering the transient impact phase and the initial stabilizing oscillation).
    
- **FSM Management:** Checks the status of the disposal button flag at the exact moment of the trigger to tag the window payload as either an addition or a removal event.
    

##|# Core 1: TinyML Inference Processing

- Stays in a blocked `vTaskDelay` state until a completed 64-sample data array is pushed into a FreeRTOS queue by Core 0.
    
- Wakes up instantly to unpack the data payload, normalize the vector, execute the 1D-CNN inference, and pass the predicted asset ID back to the main inventory loop.
    

### 3. TinyML & Inference Pipeline

- **Model Architecture:** A lightweight **1D Convolutional Neural Network (1D-CNN)**. 1D convolutions are computationally trivial for the ESP32-S3 but highly effective at extracting translation-invariant features from time-series waveforms.
    
    - **Input Shape:** $64 \times 1$ (64 scalar weight/force values).
        
    - **Layers:** Two 1D Convolutional layers (e.g., 8 and 16 filters, kernel size = 3) interleaved with Max Pooling 1D layers, terminating in a Dense layer with a Softmax output matching the number of item classes.
        
- **Optimization:** Quantized to **INT8 precision** via TensorFlow Lite Micro and accelerated using the **ESP-NN** library.
    
- **Memory Envelope:** The entire 1D-CNN model array and its `Tensor Arena` will comfortably sit well within $40\text{ KB}$ of internal SRAM, allowing the remaining memory space to be heavily utilized for data logging and system buffers.
    

### 4. State Machine & Decision Fusion Logic

The system maintains a local digital manifest in memory to track items. The physical button dictates the execution path when an inference event finishes.

|**Current System State**|**Action Taken**|**Validation Check**|**Manifest Resolution**|
|---|---|---|---|
|**`STATE_ADD`** _(Default)_|Item dropped in. Model predicts Class ID.|Asserts that absolute steady-state weight change ($\Delta W$) matches the expected static mass profile of the predicted class: $\Delta W > 0$.|Class ID added to cart inventory array.|
|**`STATE_REMOVE`** _(Button Active)_|Item lifted out. Model predicts Class ID.|Asserts that absolute steady-state weight change ($\Delta W$) matches the expected static mass profile of the predicted class: $\Delta W < 0$.|Looks up Class ID in the active inventory array and drops one instance.|

#### Anomaly Flagging

If the 1D-CNN model predicts an item (e.g., "Item A") but the final resting weight delta ($\Delta W$) does not align with Item A's static entry in the local LittleFS product configuration file within an acceptable tolerance window ($\pm 2\sigma$), the system flags a mismatch anomaly.

### Data Sourcing & Training Protocol

1. **Raw Data Harvester Firmware:** You will write a straightforward telemetry script in PlatformIO. Place an item on the load cell, trigger a sample capture, and stream the raw 64-point array as a comma-separated line directly over the USB-CDC serial port to a waiting Python script on your development machine.
    
2. **Dataset Balancing:** Collect 50–100 impact profiles per item category. Ensure you capture variations in drop height, orientation, and removal dynamics (slow lifts vs. rapid yanks) to make the model resilient.
    
3. **Training Script:** Build the 1D-CNN in Keras, convert the final model to a TensorFlow Lite file using Post-Training Quantization (PTQ), and run the final array generation through `xxd -i` to create the C header file asset.