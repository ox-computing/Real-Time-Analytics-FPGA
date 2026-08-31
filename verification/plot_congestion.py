#!/usr/bin/env python3
"""Live viewer for nextpnr router2 congestion heatmaps.

router2 (--router2-heatmap <prefix>) writes one CSV per ripup-reroute
iteration, flushed as it goes, so the routing run can be watched while it is
still in progress. The by-coordinate file is a bare matrix -- one row per chip
X, one comma-separated value per Y, no header -- which is only readable as a
picture.

This polls the directory for new iteration files and redraws. Space pauses on
the current iteration so a hot spot can be studied while the router keeps
running; releasing the pause jumps forward to whatever the latest iteration is
by then, rather than replaying the frames that were missed.

Keys:
  space        pause / resume (resume jumps to the newest iteration)
  left/right   step one iteration while paused
  home/end     jump to first / newest iteration while paused
  s            save the current frame as a PNG next to the CSVs
  q            quit

Usage (or just `make plot`, which points --dir at data/routing for you):
  python3 verification/plot_congestion.py --dir data/routing
  python3 verification/plot_congestion.py --dir data/routing --no-follow
"""

import argparse
import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

# congestion_by_coordinate is the only spatial heatmap router2 writes; the
# others (by_wiretype, by_net) are tables, not maps. The router prepends the
# user's --router2-heatmap prefix, so match on the suffix and stay
# prefix-agnostic.
PATTERN = "*congestion_by_coordinate_*.csv"
ITER_RE = re.compile(r"congestion_by_coordinate_(\d+)\.csv$")


def find_frames(directory):
    """Map iteration number -> (path, mtime) for the CURRENT routing run.

    Iteration numbering restarts at 1 on every run and the filenames are
    reused, so a directory routinely holds a mix: the run being watched, plus
    higher-numbered leftovers from a previous, longer run. Iteration number
    alone cannot tell them apart, but mtime can -- the router writes iterations
    in order, so the most recently modified file IS the current run's newest
    iteration, and anything numbered above it is a leftover to be ignored.
    """
    stamped = {}
    for path in glob.glob(os.path.join(directory, PATTERN)):
        m = ITER_RE.search(os.path.basename(path))
        if not m:
            continue
        try:
            stamped[int(m.group(1))] = (path, os.path.getmtime(path))
        except OSError:
            pass  # vanished between glob and stat
    if not stamped:
        return {}

    newest = max(stamped, key=lambda it: stamped[it][1])
    return {it: v for it, v in stamped.items() if it <= newest}


def load_frame(path):
    """Read one congestion matrix, or None if it is still being written.

    The rows are chip X and the columns chip Y, with a trailing comma on every
    line. Returned transposed so that imshow puts X on the horizontal axis.
    A file caught mid-write has short or ragged rows; treat that as not-ready
    and let the caller retry on the next poll.
    """
    try:
        with open(path) as f:
            text = f.read()
    except OSError:
        return None

    rows = []
    for line in text.splitlines():
        line = line.rstrip(",")
        if not line:
            continue
        try:
            rows.append([int(v) for v in line.split(",")])
        except ValueError:
            return None

    if not rows or len({len(r) for r in rows}) != 1:
        return None
    return np.array(rows, dtype=float).T


class Viewer:
    def __init__(self, directory, follow, interval_ms):
        self.dir = directory
        self.follow = follow
        self.frames = find_frames(directory)
        self.cache = {}
        if not self.frames:
            sys.exit(
                f"No {PATTERN} in {directory!r}.\n"
                "Run the router with --router2-heatmap <prefix> (see `make heatmap`)."
            )

        self.paused = False
        # Congestion falls over a converging run, so a per-frame colour scale
        # would rescale every redraw and make it look like nothing improved.
        # Hold the scale at the worst congestion seen so far instead.
        self.vmax = 1.0

        newest = self.newest_ready()
        if newest is None:
            sys.exit(
                f"Found {len(self.frames)} heatmap file(s) in {directory!r} but none "
                "parsed -- the router may have only just started writing the first "
                "one. Try again in a second."
            )
        self.iter = newest
        self.shown_mtime = self.frames[newest][1]
        data = self.get(newest)
        self.fig, self.ax = plt.subplots(figsize=(8, 7))
        self.im = self.ax.imshow(
            data, origin="lower", cmap="inferno", interpolation="nearest", vmin=0
        )
        self.ax.set_xlabel("chip X")
        self.ax.set_ylabel("chip Y")
        self.cbar = self.fig.colorbar(self.im, ax=self.ax)
        self.cbar.set_label("overuse (sum of curr_cong where > 1)")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw(data, newest)

        self.timer = self.fig.canvas.new_timer(interval=interval_ms)
        self.timer.add_callback(self.tick)
        self.timer.start()

    def get(self, it):
        """Parsed matrix for an iteration, or None if it is not readable yet.

        Cached against the file's mtime, not just its number: a new run reuses
        the same filenames, so a frame already held under some iteration number
        can be overwritten with entirely different content. Caching on the
        number alone would pin the viewer to the previous run's data forever.

        The newest file is routinely one the router is still flushing, so a
        parse failure is a normal transient, not an error: leave it out of the
        cache and it will be picked up on a later poll.
        """
        path, mtime = self.frames[it]
        hit = self.cache.get(it)
        if hit is not None and hit[0] == mtime:
            return hit[1]
        data = load_frame(path)
        if data is None:
            return None
        self.cache[it] = (mtime, data)
        return data

    def newest_ready(self):
        """Highest iteration that actually parses, skipping any mid-write tail."""
        for it in sorted(self.frames, reverse=True):
            if self.get(it) is not None:
                return it
        return None

    def rescan(self):
        """Re-stat the directory, dropping frames the current run has orphaned."""
        self.frames = find_frames(self.dir)
        for it in list(self.cache):
            if it not in self.frames:
                del self.cache[it]

    def draw(self, data, newest):
        self.im.set_data(data)
        self.im.set_extent((0, data.shape[1], 0, data.shape[0]))
        self.vmax = max(self.vmax, float(data.max()))
        self.im.set_clim(0, self.vmax)

        hot = int(data.max())
        total = int(data.sum())
        if self.paused:
            behind = newest - self.iter
            state = "PAUSED" + (f" -- {behind} newer" if behind else "")
        else:
            state = "live" if self.follow else "browsing"
        self.ax.set_title(
            f"router2 congestion -- iteration {self.iter}/{newest}  [{state}]\n"
            f"peak {hot} on one tile, {total} overused total"
        )
        self.fig.canvas.draw_idle()

    def show(self, it):
        if it is None or it not in self.frames:
            return
        data = self.get(it)
        if data is None:
            return  # mid-write; a later poll will pick it up
        self.iter = it
        self.shown_mtime = self.frames[it][1]
        self.draw(data, self.newest_ready())

    def tick(self):
        if self.follow:
            self.rescan()
        newest = self.newest_ready()
        if not self.paused:
            # Redraw when the newest frame is a different iteration OR when the
            # file behind the current one has been rewritten -- a restarted run
            # reuses the numbers, so an unchanged iteration number does not mean
            # unchanged data.
            if newest is not None and (
                newest != self.iter or self.frames[newest][1] != self.shown_mtime
            ):
                self.show(newest)
            return
        # Paused: hold the frame, but keep polling so the title can report how
        # far the router has run on ahead of what is on screen. The held frame
        # can be pruned or rewritten by a restarted run; if it is momentarily
        # unreadable, keep what is already on screen rather than blanking it.
        if newest is None or self.iter not in self.frames:
            return
        data = self.get(self.iter)
        if data is not None:
            self.draw(data, newest)

    def on_key(self, event):
        if event.key == " ":
            self.paused = not self.paused
            # Resuming skips the backlog: the point of the pause was to hold a
            # frame, not to queue up a replay of a router that has moved on.
            self.show(self.iter if self.paused else self.newest_ready())
        elif event.key == "right" and self.paused:
            later = [i for i in self.frames if i > self.iter]
            if later:
                self.show(min(later))
        elif event.key == "left" and self.paused:
            earlier = [i for i in self.frames if i < self.iter]
            if earlier:
                self.show(max(earlier))
        elif event.key == "home" and self.paused:
            self.show(min(self.frames))
        elif event.key == "end" and self.paused:
            self.show(self.newest_ready())
        elif event.key == "s":
            out = os.path.join(self.dir, f"congestion_iter{self.iter}.png")
            self.fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"wrote {out}")
        elif event.key == "q":
            plt.close(self.fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=".", help="directory holding the CSVs (default: .)")
    ap.add_argument(
        "--no-follow",
        action="store_true",
        help="do not poll for new iterations; browse what is already there",
    )
    ap.add_argument(
        "--interval", type=float, default=1.0, help="poll seconds (default: 1)"
    )
    args = ap.parse_args()

    Viewer(args.dir, follow=not args.no_follow, interval_ms=int(args.interval * 1000))
    plt.show()


if __name__ == "__main__":
    main()
