import ctypes
import sys
import re
from .classes import *
from .variables import *

# Pattern to match [PAD...] tokens that RKLLM outputs for unknown tokens
PAD_TOKEN_PATTERN = re.compile(r'\[PAD(\d+)\]')

# Token ID for <think> - we'll replace PAD tokens with this
# DeepSeek uses token 151935 for thinking, map to <think> marker
THINKING_TOKEN_IDS = {151935}  # Add more if discovered

# Track if we're in thinking mode
_in_thinking_block = False
_thinking_token_count = 0


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
    global _in_thinking_block, _thinking_token_count

    if state == LLMCallState.RKLLM_RUN_FINISH:
        global_status = state
        # If we were in a thinking block, close it
        if _in_thinking_block:
            closing = f"</think> ({_thinking_token_count} thinking tokens)\n"
            global_text.append(closing)
            print(closing, end='')
            _in_thinking_block = False
            _thinking_token_count = 0
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
                token_id = result.contents.token_id

                # Ensure we have bytes
                if not isinstance(text_bytes, bytes):
                    try:
                        text_bytes = bytes(text_bytes)
                    except:
                        text_bytes = b""

                # Try to decode, handling incomplete UTF-8 sequences
                try:
                    decoded_text = (split_byte_data + text_bytes).decode('utf-8')
                    split_byte_data = bytes(b"")

                    # Check for [PAD...] tokens (unknown tokens from extended vocab)
                    pad_match = PAD_TOKEN_PATTERN.match(decoded_text)
                    if pad_match:
                        pad_id = int(pad_match.group(1))
                        # These are thinking tokens
                        if not _in_thinking_block:
                            _in_thinking_block = True
                            _thinking_token_count = 0
                            opening = "<think>"
                            global_text.append(opening)
                            print(opening, end='')
                        _thinking_token_count += 1
                        # Print a dot every 100 thinking tokens to show progress
                        if _thinking_token_count % 100 == 0:
                            print(".", end='', file=sys.stderr)
                            sys.stderr.flush()
                        return

                    # If we were in thinking and now have real text, close the block
                    if _in_thinking_block:
                        closing = f"</think> ({_thinking_token_count} tokens)\n"
                        global_text.append(closing)
                        print(closing, end='')
                        _in_thinking_block = False
                        _thinking_token_count = 0

                    global_text.append(decoded_text)
                    print(decoded_text, end='')
                except UnicodeDecodeError:
                    # Incomplete UTF-8 sequence, save for next callback
                    split_byte_data += text_bytes
            else:
                # Handle case where text is None
                if split_byte_data:
                    try:
                        decoded_text = split_byte_data.decode('utf-8')

                        # Check for [PAD...] pattern
                        if PAD_TOKEN_PATTERN.match(decoded_text):
                            if not _in_thinking_block:
                                _in_thinking_block = True
                                _thinking_token_count = 0
                            _thinking_token_count += 1
                            split_byte_data = bytes(b"")
                            return

                        if _in_thinking_block:
                            closing = f"</think> ({_thinking_token_count} tokens)\n"
                            global_text.append(closing)
                            print(closing, end='')
                            _in_thinking_block = False
                            _thinking_token_count = 0

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
