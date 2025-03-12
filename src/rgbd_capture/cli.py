import click

from rgbd_capture import capture


@click.group()
def cli():
    pass


@cli.command(help="Display camera view.")
@click.option("-w", "--width", type=int, default=1280, help="frame width")
@click.option("-h", "--height", type=int, default=720, help="frame height")
@click.option("--fps", type=int, default=15, help="frames per second")
@click.option("--max-depth", type=int, default=5_000, help="maximum raw depth value")
def view(width, height, fps, max_depth):
    capture.view(frame_dimensions=(width, height), fps=fps, max_depth=max_depth)


@cli.command(help="Take a snapshot.")
@click.option("-w", "--width", type=int, default=1280, help="frame width")
@click.option("-h", "--height", type=int, default=720, help="frame height")
@click.option("--fps", type=int, default=15, help="frames per second")
@click.option("-c", "--color-filename", type=str, help="filename for the color frame")
@click.option("-d", "--depth-filename", type=str, help="filename for the depth frame")
@click.option(
    "-p", "--preview", type=bool, default=True, help="preview snapshot before saving"
)
@click.option(
    "--frames-to-skip",
    type=int,
    default=10,
    help="number of frames to skip before taking a snapshot so that autoexposure can settle",
)
def snap(color_filename, depth_filename, width, height, fps, preview, frames_to_skip):
    capture.snap(
        color_filename,
        depth_filename,
        frame_dimensions=(width, height),
        fps=fps,
        frames_to_skip=frames_to_skip,
        preview=preview,
    )


@cli.command(help="Take a burst of snapshots.")
@click.option("-w", "--width", type=int, default=1280, help="frame width")
@click.option("-h", "--height", type=int, default=720, help="frame height")
@click.option("--fps", type=int, default=15, help="frames per second")
@click.option(
    "-d",
    "--burst-dir",
    type=str,
    default=".",
    help="directory for the sequence of images",
)
@click.option(
    "--frames-to-skip",
    type=int,
    default=10,
    help="number of frames to skip before taking a snapshot so that autoexposure can settle",
)
@click.option(
    "-n",
    "--frames-to-capture",
    type=int,
    default=10,
    help="number of frames to save in the burst",
)
def burst(burst_dir, width, height, fps, frames_to_skip, frames_to_capture):
    capture.burst(
        burst_dir=burst_dir,
        frame_dimensions=(width, height),
        fps=fps,
        frames_to_skip=frames_to_skip,
        frames_to_capture=frames_to_capture,
    )


@cli.command(help="Record a video.")
@click.option("-w", "--width", type=int, default=1280, help="frame width")
@click.option("-h", "--height", type=int, default=720, help="frame height")
@click.option("--fps", type=int, default=15, help="frames per second")
@click.option("-c", "--color-filename", type=str, help="filename for the color frame")
@click.option("-d", "--depth-filename", type=str, help="filename for the depth frame")
def record():
    pass


@cli.command(help="Print or save camera info.")
@click.option(
    "-f", "--filename", type=str, help="If given, write camera information to a file."
)
@click.option(
    "-w",
    "--width",
    type=int,
    default=1280,
    help="frame width, since intrinsics change with the frame dimension",
)
@click.option(
    "-h",
    "--height",
    type=int,
    default=720,
    help="frame height, since intrinsics change with the frame dimension",
)
def info(filename, width, height):
    capture.info((width, height), filename)


def main():
    cli(auto_envvar_prefix="RGBD")


if __name__ == "__main__":
    main()
