# Architecture Documentation

This document provides visual architecture documentation for RKLLama - the LLM inference server for Rockchip RK3588/RK3576 NPU.

## Table of Contents

1. [System Overview](#system-overview)
2. [Network Topology](#network-topology)
3. [Component Interaction](#component-interaction)
4. [Data Flow](#data-flow)
5. [Token Streaming Architecture](#token-streaming-architecture)
6. [Model Loading Pipeline](#model-loading-pipeline)

---

## System Overview

```mermaid
graph TB
    subgraph "Client Layer"
        CLI[rkllama CLI<br/>client.py]
        HTTP[HTTP Clients<br/>curl, Python requests]
        EXO[Exo Framework<br/>Distributed Inference]
    end

    subgraph "Server Layer"
        Flask[Flask REST API<br/>server.py:8080]
        Routes[API Endpoints]
        Lock[Thread Lock<br/>Single Model Access]
    end

    subgraph "Processing Layer"
        Process[Request Processor<br/>src/process.py]
        Tokenizer[HuggingFace<br/>AutoTokenizer]
        ChatTemplate[Chat Template<br/>Jinja2]
    end

    subgraph "RKLLM Runtime"
        RKLLM[RKLLM Wrapper<br/>src/rkllm.py]
        CTypes[ctypes Bindings<br/>src/classes.py]
        Callback[Token Callback<br/>src/callback.py]
    end

    subgraph "Hardware Layer"
        LibRKLLM[librkllmrt.so<br/>Native C Library]
        NPU[RK3588 NPU<br/>6 TOPS INT8]
        Memory[LPDDR5 Memory<br/>Model Cache]
    end

    CLI --> Flask
    HTTP --> Flask
    EXO --> Flask

    Flask --> Routes
    Routes --> Lock
    Lock --> Process

    Process --> Tokenizer
    Tokenizer --> ChatTemplate
    ChatTemplate --> RKLLM

    RKLLM --> CTypes
    CTypes --> LibRKLLM
    LibRKLLM --> NPU
    LibRKLLM --> Memory

    LibRKLLM --> Callback
    Callback --> Process
```

---

## Network Topology

```mermaid
graph TB
    subgraph "External Clients"
        Browser[Web Browser]
        Script[Python Scripts]
        ExoNode[Exo Node<br/>Distributed Inference]
    end

    subgraph "RK3588 Device"
        subgraph "Network Interface"
            ETH[Ethernet<br/>1Gbps]
        end

        subgraph "RKLLama Server"
            Flask[Flask Server<br/>0.0.0.0:8080]
            Threaded[Threaded Mode<br/>Multi-Connection]
        end

        subgraph "API Endpoints"
            GET_Root["GET /<br/>Health Check"]
            GET_Models["GET /models<br/>List Models"]
            POST_Load["POST /load_model<br/>Load to NPU"]
            POST_Unload["POST /unload_model<br/>Free NPU"]
            GET_Current["GET /current_model<br/>Active Model"]
            POST_Generate["POST /generate<br/>Inference"]
            POST_Pull["POST /pull<br/>HuggingFace DL"]
            DELETE_RM["DELETE /rm<br/>Delete Model"]
        end

        subgraph "Model Storage"
            ModelDir["~/RKLLAMA/models/"]
            ModelFiles["*.rkllm Files"]
            Modelfile["Modelfile Config"]
        end

        subgraph "NPU Hardware"
            NPU[NPU Cores<br/>3x2 TOPS]
            DMA[DMA Engine]
        end
    end

    Browser --> ETH
    Script --> ETH
    ExoNode --> ETH

    ETH --> Flask
    Flask --> Threaded

    Threaded --> GET_Root
    Threaded --> GET_Models
    Threaded --> POST_Load
    Threaded --> POST_Unload
    Threaded --> GET_Current
    Threaded --> POST_Generate
    Threaded --> POST_Pull
    Threaded --> DELETE_RM

    POST_Load --> ModelDir
    ModelDir --> ModelFiles
    ModelDir --> Modelfile
    ModelFiles --> NPU
```

---

## Component Interaction

```mermaid
graph LR
    subgraph "server.py"
        App[Flask App]
        Routes[Route Handlers]
        ModelState[Global Model State]
    end

    subgraph "src/process.py"
        RequestHandler[Request Handler]
        MessageFormatter[Message Formatter]
        StreamHandler[Stream Handler]
    end

    subgraph "src/rkllm.py"
        RKLLMClass[RKLLM Class]
        Init[__init__<br/>Load Model]
        Run[run<br/>Token Inference]
        RunPrompt[run_prompt<br/>Text Inference]
        Release[release<br/>Cleanup]
    end

    subgraph "src/classes.py"
        RKLLMParam[RKLLMParam Struct]
        RKLLMInput[RKLLMInput Union]
        RKLLMResult[RKLLMResult Struct]
        RKLLMExtend[RKLLMExtendParam]
    end

    subgraph "src/callback.py"
        CallbackFunc[Callback Function]
        UTF8Buffer[UTF-8 Buffer]
        TokenQueue[Token Queue]
    end

    subgraph "src/variables.py"
        GlobalText[global_text List]
        Verrou[Threading Lock]
        ModelID[model_id State]
    end

    subgraph "src/special_tokens.py"
        DeepSeek[DeepSeek Tokens]
        Llama[Llama Tokens]
        Qwen[Qwen Tokens]
    end

    App --> Routes
    Routes --> ModelState
    Routes --> RequestHandler

    RequestHandler --> MessageFormatter
    MessageFormatter --> RKLLMClass
    RequestHandler --> StreamHandler

    RKLLMClass --> Init
    RKLLMClass --> Run
    RKLLMClass --> RunPrompt
    RKLLMClass --> Release

    Init --> RKLLMParam
    Run --> RKLLMInput
    Run --> CallbackFunc

    CallbackFunc --> RKLLMResult
    CallbackFunc --> UTF8Buffer
    CallbackFunc --> TokenQueue
    CallbackFunc --> GlobalText

    TokenQueue --> StreamHandler

    Init --> RKLLMExtend
    MessageFormatter --> DeepSeek
    MessageFormatter --> Llama
    MessageFormatter --> Qwen

    Run --> Verrou
    ModelState --> ModelID
```

---

## Data Flow

### Inference Request Flow

```mermaid
sequenceDiagram
    participant Client
    participant Flask as Flask Server
    participant Process as process.py
    participant Tokenizer as HuggingFace
    participant RKLLM as rkllm.py
    participant Native as librkllmrt.so
    participant Callback as callback.py

    Client->>Flask: POST /generate
    Note over Flask: JSON: model, messages, stream

    Flask->>Flask: Acquire thread lock
    Flask->>Process: Handle request

    Process->>Tokenizer: Load AutoTokenizer
    Tokenizer-->>Process: Tokenizer instance

    Process->>Tokenizer: apply_chat_template(messages)
    Tokenizer-->>Process: Formatted prompt

    Process->>Tokenizer: encode(prompt)
    Tokenizer-->>Process: Token IDs

    Process->>RKLLM: run(token_ids)
    RKLLM->>Native: rkllm_run(ctx, input, infer_param, callback)

    loop Token Generation
        Native->>Callback: callback(result, state)
        Callback->>Callback: Decode UTF-8
        Callback->>Process: Append to global_text

        alt Streaming Mode
            Process-->>Client: SSE: {"token": "..."}
        end
    end

    Native-->>RKLLM: Generation complete
    RKLLM-->>Process: Return

    alt Batch Mode
        Process-->>Client: JSON response
    end

    Flask->>Flask: Release thread lock
```

### Model Loading Flow

```mermaid
sequenceDiagram
    participant Client
    participant Flask as Flask Server
    participant RKLLM as rkllm.py
    participant Classes as classes.py
    participant Native as librkllmrt.so
    participant NPU

    Client->>Flask: POST /load_model
    Note over Flask: {"model_name": "deepseek-1.5b"}

    Flask->>Flask: Check ~/RKLLAMA/models/{name}/

    Flask->>Flask: Parse Modelfile
    Note over Flask: FROM, HUGGINGFACE_PATH,<br/>SYSTEM, TEMPERATURE

    Flask->>RKLLM: RKLLM(model_path, ...)

    RKLLM->>Classes: Create RKLLMParam
    RKLLM->>Classes: Create RKLLMExtendParam
    Note over Classes: CPU cores 4-7 (big cores)

    RKLLM->>Native: rkllm_createDefaultParam()
    Native-->>RKLLM: Default params

    RKLLM->>Native: rkllm_init(ctx, param, callback)

    Native->>NPU: Load model weights
    NPU-->>Native: Model ready

    Native-->>RKLLM: Context handle

    RKLLM-->>Flask: RKLLM instance
    Flask-->>Client: {"status": "loaded"}
```

---

## Token Streaming Architecture

```mermaid
graph TB
    subgraph "Input Processing"
        Messages[Chat Messages]
        Template[Chat Template]
        Tokens[Token IDs]
    end

    subgraph "Native Inference"
        RKLLM_Run[rkllm_run]
        NPU_Compute[NPU Computation]
        Token_Gen[Token Generation]
    end

    subgraph "Callback System"
        C_Callback[C Callback Function]
        UTF8_Handler[UTF-8 Handler]
        Incomplete_Buffer[Incomplete Byte Buffer]
    end

    subgraph "Output Modes"
        SSE[Server-Sent Events]
        JSON_Batch[JSON Batch Response]
    end

    subgraph "Global State"
        global_text[global_text List]
        global_status[global_status Flag]
    end

    Messages --> Template
    Template --> Tokens
    Tokens --> RKLLM_Run

    RKLLM_Run --> NPU_Compute
    NPU_Compute --> Token_Gen
    Token_Gen --> C_Callback

    C_Callback --> UTF8_Handler
    UTF8_Handler --> Incomplete_Buffer
    Incomplete_Buffer --> UTF8_Handler

    UTF8_Handler --> global_text
    C_Callback --> global_status

    global_text --> SSE
    global_text --> JSON_Batch
    global_status --> SSE
    global_status --> JSON_Batch
```

### Callback State Machine

```mermaid
stateDiagram-v2
    [*] --> RKLLM_RUN_NORMAL: Token generated
    RKLLM_RUN_NORMAL --> RKLLM_RUN_NORMAL: More tokens
    RKLLM_RUN_NORMAL --> RKLLM_RUN_FINISH: EOS token
    RKLLM_RUN_NORMAL --> RKLLM_RUN_ERROR: Error occurred
    RKLLM_RUN_NORMAL --> RKLLM_RUN_WAITING: Context switch

    RKLLM_RUN_FINISH --> [*]: Complete
    RKLLM_RUN_ERROR --> [*]: Error response
    RKLLM_RUN_WAITING --> RKLLM_RUN_NORMAL: Resume

    state RKLLM_RUN_NORMAL {
        [*] --> DecodeUTF8
        DecodeUTF8 --> CheckComplete: Byte received
        CheckComplete --> AppendGlobalText: Valid UTF-8
        CheckComplete --> BufferByte: Incomplete sequence
        BufferByte --> DecodeUTF8: Wait for more
        AppendGlobalText --> [*]
    }
```

---

## Model Loading Pipeline

```mermaid
flowchart TD
    subgraph "Model Discovery"
        A[Scan ~/RKLLAMA/models/] --> B{Model Found?}
        B -->|No| C[Return Error]
        B -->|Yes| D[Read Modelfile]
    end

    subgraph "Configuration Parsing"
        D --> E[Parse FROM field]
        E --> F[Get .rkllm path]
        D --> G[Parse HUGGINGFACE_PATH]
        G --> H[Tokenizer source]
        D --> I[Parse SYSTEM prompt]
        D --> J[Parse TEMPERATURE]
    end

    subgraph "RKLLM Initialization"
        F --> K[Create RKLLMParam]
        J --> K
        K --> L[Set CPU affinity<br/>Cores 4-7]
        L --> M[Create context]
        M --> N[Load to NPU]
    end

    subgraph "Tokenizer Setup"
        H --> O[Load AutoTokenizer]
        O --> P[Cache tokenizer]
    end

    subgraph "Ready State"
        N --> Q[Model Loaded]
        P --> Q
        Q --> R[Ready for inference]
    end
```

---

## Configuration Reference

### Modelfile Format

```
FROM="model-name.rkllm"
HUGGINGFACE_PATH="org/repo-name"
SYSTEM="You are a helpful assistant."
TEMPERATURE=0.8
MAX_NEW_TOKENS=2048
```

### API Response Formats

**Streaming (SSE):**
```json
{"token": "Hello", "thinking": false}
{"token": " world", "thinking": false}
{"token": "", "finish": true, "tokens_per_second": 15.2}
```

**Batch:**
```json
{
  "response": "Hello world",
  "tokens_generated": 2,
  "tokens_per_second": 15.2
}
```

---

## File Structure

```
rkllama/
├── server.py              # Flask REST API server
├── client.py              # CLI client
├── rkllama.ini            # Configuration file
├── lib/
│   └── librkllmrt.so      # Native RKLLM runtime
├── src/
│   ├── rkllm.py           # RKLLM wrapper class
│   ├── classes.py         # ctypes struct definitions
│   ├── callback.py        # Token streaming callback
│   ├── process.py         # Request processing
│   ├── special_tokens.py  # Model-specific tokens
│   └── variables.py       # Global state
├── models/                # Model storage directory
│   └── {model_name}/
│       ├── Modelfile      # Model configuration
│       └── *.rkllm        # Model weights
└── documentation/
    └── api/               # API documentation
```
