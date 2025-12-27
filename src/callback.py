import ctypes
import sys
from .classes import *
from .variables import *


def callback_impl(result, userdata, state):
    """
    Callback function for RKLLM inference results.
    Updated for RKLLM runtime 1.2.3.

    Args:
        result: Pointer to RKLLMResult structure
        userdata: User data pointer (unused)
        state: LLMCallState indicating the current state
    """
    global split_byte_data, global_text, global_status

    if state == LLMCallState.RKLLM_RUN_FINISH:
        global_status = state
        print("\n")
        sys.stdout.flush()

    elif state == LLMCallState.RKLLM_RUN_ERROR:
        global_status = state
        print("Execution error")
        sys.stdout.flush()

    elif state == LLMCallState.RKLLM_RUN_NORMAL:
        # Save the output token text and execution state
        global_status = state

        try:
            # Check if result and its contents are valid
            if result and result.contents and result.contents.text:
                text_bytes = result.contents.text

                # Ensure we have bytes
                if not isinstance(text_bytes, bytes):
                    try:
                        text_bytes = bytes(text_bytes)
                    except:
                        text_bytes = b""

                # Try to decode, handling incomplete UTF-8 sequences
                try:
                    decoded_text = (split_byte_data + text_bytes).decode('utf-8')
                    global_text.append(decoded_text)
                    print(decoded_text, end='')
                    split_byte_data = bytes(b"")
                except UnicodeDecodeError:
                    # Incomplete UTF-8 sequence, save for next callback
                    split_byte_data += text_bytes
            else:
                # Handle case where text is None
                if split_byte_data:
                    try:
                        decoded_text = split_byte_data.decode('utf-8')
                        global_text.append(decoded_text)
                        print(decoded_text, end='')
                        split_byte_data = bytes(b"")
                    except UnicodeDecodeError:
                        # Still incomplete, keep for next time
                        pass

        except Exception as e:
            print(f"\nError processing callback: {str(e)}", end='')

        sys.stdout.flush()

    elif state == LLMCallState.RKLLM_RUN_WAITING:
        # Model is waiting for more input (used in chat mode)
        global_status = state
        sys.stdout.flush()

    else:
        # Unknown state
        global_status = state
        sys.stdout.flush()
