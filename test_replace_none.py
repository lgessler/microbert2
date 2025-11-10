#!/usr/bin/env python
"""
Script to check if Tango's _replace_none function is being called.
This patches the function to add logging, then runs the full tango pipeline.

Usage:
    python test_replace_none.py [config_file]

Example:
    python test_replace_none.py configs/microbert_tamil.jsonnet
"""
import sys
import os
import traceback

# Track calls globally
call_tracker = {'count': 0, 'calls': []}


def patch_replace_none():
    """Patch _replace_none before any tango imports"""
    try:
        from tango.common.params import _replace_none
        import tango.common.params

        original_func = _replace_none

        def traced_replace_none(params):
            """Wrapper that logs when _replace_none is called"""
            call_tracker['count'] += 1
            call_num = call_tracker['count']

            print(f"\n{'='*80}")
            print(f"🔍 CALL #{call_num} to _replace_none()")
            print(f"{'='*80}")
            print(f"Input type: {type(params)}")

            # Show preview of input
            params_str = str(params)
            if len(params_str) > 500:
                print(f"Input value (first 500 chars): {params_str[:500]}...")
            else:
                print(f"Input value: {params_str}")

            # Show relevant stack trace
            print(f"\nCall stack (relevant frames):")
            stack = traceback.format_stack()[:-1]
            for frame in stack:
                if '/tango/' in frame or '/microbert2/' in frame or 'jsonnet' in frame.lower():
                    # Clean up the frame output
                    lines = frame.strip().split('\n')
                    for line in lines:
                        print(f"  {line}")

            # Call original function
            result = original_func(params)

            # Show result
            result_str = str(result)
            if len(result_str) > 500:
                print(f"\nOutput value (first 500 chars): {result_str[:500]}...")
            else:
                print(f"\nOutput value: {result_str}")
            print(f"Output type: {type(result)}")
            print(f"{'='*80}\n")

            # Track this call
            call_tracker['calls'].append({
                'num': call_num,
                'input_type': type(params).__name__,
                'output_type': type(result).__name__,
            })

            return result

        # Replace the function
        tango.common.params._replace_none = traced_replace_none
        print("✅ Successfully patched _replace_none")
        return True

    except ImportError as e:
        print(f"❌ Could not import _replace_none: {e}")
        print("\nTrying to locate it...")

        try:
            import tango
            tango_path = os.path.dirname(tango.__file__)
            print(f"\nTango installed at: {tango_path}")
            print("\nYou can search for _replace_none with:")
            print(f"  grep -r 'def _replace_none' {tango_path}")
        except:
            pass

        return False


def run_tango_with_tracing(config_path):
    """Run tango with the patched _replace_none"""
    print("="*80)
    print("TANGO PIPELINE RUN WITH _replace_none TRACING")
    print("="*80)
    print(f"Config: {config_path}\n")

    # Patch before running
    if not patch_replace_none():
        print("\n❌ Failed to patch _replace_none, exiting")
        sys.exit(1)

    print("\nStarting tango run...")
    print("(This will run the full pipeline - press Ctrl+C to stop)\n")

    try:
        # Import tango CLI and run
        from tango.cli import main as tango_main

        # Set up arguments for tango run
        original_argv = sys.argv
        sys.argv = ['tango', 'run', config_path, '--include-package', 'microbert2']

        # Run the pipeline
        tango_main()

        # Restore argv
        sys.argv = original_argv

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
    except SystemExit:
        # Tango may call sys.exit, which is normal
        pass
    except Exception as e:
        print(f"\n\n❌ Error during tango run: {e}")
        traceback.print_exc()
    finally:
        # Print summary
        print("\n\n" + "="*80)
        print("TRACING SUMMARY")
        print("="*80)
        print(f"Total calls to _replace_none: {call_tracker['count']}")

        if call_tracker['count'] > 0:
            print("\n✅ YES! _replace_none WAS CALLED during the pipeline run")
            print(f"\nTotal calls: {call_tracker['count']}")

            if call_tracker['calls']:
                print("\nCall summary:")
                for call in call_tracker['calls'][:20]:  # Show first 20
                    print(f"  Call #{call['num']}: {call['input_type']} -> {call['output_type']}")

                if len(call_tracker['calls']) > 20:
                    print(f"  ... and {len(call_tracker['calls']) - 20} more calls")
        else:
            print("\n❌ NO, _replace_none WAS NOT CALLED during the pipeline run")

        print("="*80)


if __name__ == "__main__":
    # Get config path from command line or use default
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "configs/microbert_tamil.jsonnet"

    # Check if config exists
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        print(f"\nUsage: python {sys.argv[0]} [config_file]")
        print(f"Example: python {sys.argv[0]} configs/microbert_tamil.jsonnet")
        sys.exit(1)

    # Run with tracing
    run_tango_with_tracing(config_path)
