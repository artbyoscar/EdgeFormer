#!/bin/bash
# Fail if any specified file lacks a trailing newline.

ret=0
for f in "$@"; do
    [ -f "$f" ] || continue
    # skip binary files
    if grep -Iq . "$f" 2>/dev/null; then
        # get ASCII code of last byte
        last_byte=$(tail -c1 "$f" | od -An -t u1 | tr -d ' \n')
        if [ -n "$last_byte" ] && [ "$last_byte" != "10" ]; then
            echo "$f: missing trailing newline"
            ret=1
        fi
    fi
done
exit $ret
