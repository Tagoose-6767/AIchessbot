#!/usr/bin/env bash
# Downloads a GM-level Polyglot opening book into chess_engine/books/.
# The .bin file is gitignored (binary, ~340 KB) so each environment fetches it
# locally. After download, point the engine at the file via UCI:
#
#   setoption name Book value books/gm2600.bin
#
# Or set the OPENING_BOOK env var before launching the Python engine.
#
# Default book is gm2600.bin (Pascal Georges, originally bundled with Scid vs
# PC -- positions selected from games where both players >= 2600 Elo). Pass an
# alternate book name as the first argument; supported aliases below.

set -eu

cd "$(dirname "$0")/.."
mkdir -p books

case "${1:-gm2600}" in
    gm2600)
        url="https://raw.githubusercontent.com/DannyStoll1/chess-opening-prep/master/gm2600.bin"
        out="books/gm2600.bin"
        ;;
    elo2400)
        url="https://raw.githubusercontent.com/DannyStoll1/chess-opening-prep/master/Elo2400.bin"
        out="books/Elo2400.bin"
        ;;
    dec2015)
        # Lichess-derived; deeper / more variety than the GM-curated books.
        url="https://raw.githubusercontent.com/DannyStoll1/chess-opening-prep/master/dec2015.bin"
        out="books/dec2015.bin"
        ;;
    *)
        echo "unknown book: $1 (try gm2600 / elo2400 / dec2015)" >&2
        exit 1
        ;;
esac

echo "Fetching $out from $url ..."
curl -sSL --max-time 120 -o "$out" "$url" \
     -w "  http=%{http_code} size=%{size_download} type=%{content_type}\n"

# Polyglot books are arrays of 16-byte entries: a real one is at least many KB
# and its content-type is application/octet-stream. A 404 page is a few KB of
# text/html, so file the response to fail loudly when the URL went away.
size=$(wc -c < "$out")
if [ "$size" -lt 10000 ]; then
    echo "ERROR: downloaded file is only $size bytes -- likely a 404 page." >&2
    rm -f "$out"
    exit 1
fi

echo "Saved $out ($size bytes). Load via:"
echo "  setoption name Book value $out"
