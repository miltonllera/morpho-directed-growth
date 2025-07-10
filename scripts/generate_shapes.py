from functools import partial
import json
import os
import os.path as osp
from argparse import ArgumentParser
from itertools import product
from typing import NamedTuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from matplotlib.transforms import Affine2D


class DevState(NamedTuple):
    start: tuple[float, float]
    direction: tuple[float, float]


class NoOp(NamedTuple):
    def __str__(self) -> str:
        return 'noop'


class RectangleType(NamedTuple):
    height: float
    width: float
    color: tuple[float, float, float]

    def __str__(self):
        return 'rectangle'


class CircleType(NamedTuple):
    radius: float
    rotation: float
    color: tuple[float, float, float]

    def __str__(self):
        return 'circle'


class TriangleType(NamedTuple):
    side: float
    color: tuple[float, float, float]

    def __str__(self):
        return 'triangle'


def normalize(dx, dy):
    length = np.hypot(dx, dy)
    return dx / length, dy / length


def rotate(x, y, angle, about=(0, 0)):
    radians =  angle * (np.pi / 180)
    x, y = x - about[0], y - about[1]

    nx = x * np.cos(radians) - y * np.sin(radians)
    ny = x * np.sin(radians) + y * np.cos(radians)

    return nx + about[0], ny + about[1]


def radians_to_vector(radians):
    return np.sin(radians), np.cos(radians)


def translate(x, y, dx, dy):
    return x + dx, y + dy


def patch_centroid(patch):
    path = patch.get_path()
    trans = patch.get_transform()
    verts = trans.transform(path.vertices)
    return np.mean(verts, axis=0)


def add_rectangle(dev_state: DevState, rectangle_type: RectangleType, plot_attachments=False):
    height, width, color = rectangle_type
    (x_start, y_start), (dx, dy) = dev_state
    dx, dy = normalize(dx, dy)

    dx_90, dy_90 = rotate(dx, dy, -90)   # compute perpendicular direction

    # print(x_start, y_start)

    # Center the rectangle along the direction
    lower_left_x = x_start - width * dx_90 / 2
    lower_left_y = y_start - width * dy_90 / 2

    # print(lower_left_x, lower_left_y)
    # plt.scatter([lower_left_x], [lower_left_y], marker='o')

    # correction angle: to make the square face in the right direction we must rotate it along the
    # bottom left corner until the side is parallel to (dx_90, dy_90), then we need to rotate it a
    # further 90 degrees.
    correction_angle = np.degrees((np.arctan2(dy_90, dx_90) - np.arctan2(1.0, 0.0)))

    rect = patches.Rectangle(
        (lower_left_x, lower_left_y),
        width, height,
        angle=correction_angle + 90,
        # rotation_point=(x_start, y_start),
        edgecolor='none',
        facecolor=color,
        alpha=1.0
    )

    center_point = (x_start + dx * height / 2, y_start + dy * height / 2)
    next_start_point = (x_start + dx * height, y_start + dy * height)

    if plot_attachments:
        plt.scatter(next_start_point[0], next_start_point[1], color=(1.0, 0.0, 0.0))

    # Example attachment points: corners of rectangle
    attachment_points = [
        DevState(
            (center_point[0] + width * dx_90 / 2, center_point[1] + height * dy_90 / 2),
            (dx_90, dy_90),
        ),
        DevState(
            (center_point[0] - width * dx_90 / 2, center_point[1] - height * dy_90 / 2),
            (-dx_90, -dy_90),
        )
    ]

    if plot_attachments:
        plt.scatter(
            [attachment_points[0].start[0], attachment_points[1].start[0]],
            [attachment_points[0].start[1], attachment_points[1].start[1]],
            color=(0, 0, 1)
        )
    next_dev_state = DevState(next_start_point, (dx, dy))

    return next_dev_state, rect, attachment_points


def add_circle(dev_state: DevState, circle_type: CircleType, plot_attachments=False):
    (x_start, y_start), (dx, dy) = dev_state
    radius, _, color = circle_type

    dx, dy = normalize(dx, dy)

    circle = patches.Circle(
        (x_start + dx * radius, y_start + dy * radius),
        radius,
        edgecolor='none',
        facecolor=color,
        alpha=1.0
    )

    # Next point is forward by two radii
    next_start_point = (x_start + 2 * radius * dx, y_start + 2 * radius * dy)
    next_dev_state = DevState(next_start_point, (dx, dy))

    if plot_attachments:
        plt.scatter(next_start_point[0], next_start_point[1], color=(1.0, 0.0, 0.0))

    center = circle.center
    dx_90, dy_90 = rotate(dx, dy, 90)

    # create attachment_points for branches
    attachment_points = [
        (center[0] + radius * dx_90, center[1] + radius * dy_90),
        (center[0] - radius * dx_90, center[1] - radius * dy_90),
    ]

    attachment_dev_directions = [(dx_90, dy_90), (-dx_90, -dy_90) ]

    if plot_attachments:
        plt.scatter(
            [attachment_points[0][0], attachment_points[1][0]],
            [attachment_points[0][1], attachment_points[1][1]],
            color=(0, 0, 1)
        )

    branch_dev_starts = [
        DevState(start, dir) for (start, dir) in zip(attachment_points, attachment_dev_directions)
    ]

    return next_dev_state, circle, branch_dev_starts


def add_triangle(dev_state: DevState, triangle_type: TriangleType):
    (x_start, y_start), (dx, dy) = dev_state
    side, color = triangle_type

    dx, dy = normalize(dx, dy)
    h = np.sqrt(3) / 2 * side  # height of equilateral triangle

    # Base center at start, tip in direction
    base_left = (x_start - side/2 * dy, y_start + side/2 * dx)
    base_right = (x_start + side/2 * dy, y_start - side/2 * dx)
    tip = (x_start + h * dx, y_start + h * dy)

    triangle = patches.Polygon(
        [base_left, base_right, tip],
        closed=True,
        edgecolor='none',
        facecolor=color,
    )

    next_start_point = (x_start + h * dx, y_start + h * dy)
    attachment_points = []  # For now, this is a terminal structure

    return next_start_point, triangle, attachment_points


build_map = {
    'triangle': add_triangle,
    'circle': add_circle,
    'rectangle': add_rectangle,
    'noop': lambda dev_state, *_: (dev_state, None, []),
}


def plot_component(component):
    state = DevState((0.0, 0.0), (0.0, 1.0))
    _, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1, 2.0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.add_patch(build_map[str(component)](state, component, plot_attachments=True)[1])


def plot_level_components(component_list, prefix, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    for shape in component_list:
        if isinstance(shape, NoOp):
            continue
        plot_component(shape)
        plt.gcf().savefig(osp.join(save_folder, f"{prefix}_{str(shape)}"))
        plt.close(plt.gcf())


def create_branch(dev_state, level_shapes):
    patches, next_level_branches = [], []
    for shape in level_shapes:
        dev_state, patch, attachment_points = build_map[str(shape)](dev_state, shape)
        patches.append(patch)
        next_level_branches.extend(attachment_points)

    return patches, next_level_branches


def create_shape(top_level_shapes, bottom_level_shapes):
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0, 0, 0, 0))  # fully transparent

    ax.axis('off')

    state = DevState((0.0, 0.0), (0.0, 1.0))

    top_level_patches, branch_start_points = create_branch(state, top_level_shapes)
    top_level_patches = list(filter(lambda p: p is not None, top_level_patches))

    geometric_center = np.sum(
        [patch_centroid(p) for p in top_level_patches], axis=0
    ) / len(top_level_patches)

    # center the patches by shifting the axis limits
    max_size = len(top_level_shapes)
    ax.set_xlim(geometric_center[0] - 0.5 * max_size, geometric_center[0] + 0.5 * max_size )
    ax.set_ylim(geometric_center[1] - 0.5 * max_size, geometric_center[1] + 0.5 * max_size)

    for p in top_level_patches:
        ax.add_patch(p)

    for b in branch_start_points:
        branch_patches, _ = create_branch(b, bottom_level_shapes)
        for p in branch_patches:
            if p is not None:
                ax.add_patch(p)

    return fig


def is_valid(shapes, allow_empty=False):
    noops = [i for (i, s) in enumerate(shapes) if s == 0]
    first_noop = noops[0] if len(noops) > 0 else len(shapes)
    return (first_noop > 0 or allow_empty) and (first_noop + len(noops)) >= len(shapes)


def main(max_body_length, max_branch_length, save_folder):

    # x, y = 0.0, 0.0
    # fig, ax = plt.subplots(figsize=(8, 8))

    # state = DevState((0.0, 0.0), (0.0, 1.0))

    os.makedirs(save_folder, exist_ok=True)

    top_level_shapes = [
        NoOp(),
        CircleType(radius=0.5, rotation=0.0, color=(1.0, 0.0, 0.0)),
        RectangleType(height=1.0, width=1.0, color=(0.0, 1.0, 0.0)),
    ]

    bottom_level_shapes = [
        NoOp(),
        RectangleType(height=0.5, width=0.25, color=(0.0, 1.0, 1.0)),
        CircleType(radius=0.25, rotation=0.5, color=(1.0, 1.0, 0.0)),
        # TriangleType(0.2)
    ]

    plot_level_components(top_level_shapes, "top", osp.join(save_folder, "components"))
    plot_level_components(bottom_level_shapes, "bottom", osp.join(save_folder, "components"))

    top_level_sequences = list(filter(
        is_valid, product(range(len(top_level_shapes)), repeat=max_body_length)
    ))

    bottom_level_sequences = list(filter(
        partial(is_valid, allow_empty=True),
        product(range(len(top_level_shapes)), repeat=max_branch_length)
    ))


    for i, top_level_idx in enumerate(top_level_sequences):
        body_shapes = [top_level_shapes[i] for i in top_level_idx]

        for j, bottom_levl_idx in enumerate(bottom_level_sequences):
            branch_shapes = [bottom_level_shapes[i] for i in bottom_levl_idx]

            fig = create_shape(body_shapes, branch_shapes)

            fig.savefig(
                osp.join(save_folder, f"{i * len(top_level_sequences) + j}.png"), transparent=True
            )

            # for testing
            # plt.plot()
            # plt.show()
            # for testing

            plt.close(fig)

    with open(osp.join(save_folder, "metadata.json"), mode='w') as f:
        json.dump({
            'max_body_length': max_body_length,
            'max_branch_length': max_branch_length,
            'body_components': [str(s) for s in top_level_shapes],
            'branch_components': [str(s) for s in bottom_level_shapes],
        }, f)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--max_body_length", type=int)
    parser.add_argument("--max_branch_length", type=int)
    parser.add_argument("--save_folder", type=str)

    args = parser.parse_args()

    main(**vars(args))

    # top_level_shapes = [
    #     NoOp(),
    #     CircleType(radius=0.5, rotation=0.0),
    #     RectangleType(height=1.0, width=1.0),
    # ]

    # bottom_level_shapes = [
    #     NoOp(),
    #     RectangleType(height=0.5, width=0.25),
    #     CircleType(radius=0.25, rotation=0.5),
    #     # TriangleType(0.2)
    # ]

    # set seed
    # fig, ax = plt.subplots(figsize=(8, 8))
    # state = DevState((0.0, 0.0), (0.0, 1.0))
    # ax.set_xlim(-3, 3)
    # ax.set_ylim(-1, 5)

    # # state, circle, attachment_points = add_circle(state, top_level_shapes[1])
    # # ax.add_patch(circle)
    # state, rectangle, attachment_points = add_rectangle(state, top_level_shapes[2])
    # # print(attachment_points)
    # # exit()
    # plt.scatter(
    #     [attachment_points[0].start[0], attachment_points[1].start[0]],
    #     [attachment_points[0].start[1], attachment_points[1].start[1]], marker='o'
    # )
    # ax.add_patch(rectangle)

    # state, rectangle, _ = add_rectangle(attachment_points[0], bottom_level_shapes[1])
    # ax.add_patch(rectangle)

    # state, circle, attachment_points = add_circle(attachment_points[1], bottom_level_shapes[2])
    # ax.add_patch(circle)

    # state, rectangle, _ = add_rectangle(state, top_level_shapes[1])
    # ax.add_patch(rectangle)

    # # next_start_point = next_start_point[0] / 2, next_start_point[1] / 2
    # next_start_point = rotate(*next_start_point, 45, (next_start_point[0] / 2, next_start_point[1] / 2))

    # # visualize next start point
    # plt.scatter([next_start_point[0]], [next_start_point[1]], marker='x')

    # square, next_start_point, _ = add_rectangle(*next_start_point, -1.0, 1.0, width=1.0, height=1.0)
    # ax.add_patch(square)

    # circle, next_start_point, _ = add_circle(*next_start_point, -1.0, 1.0, 0.5)
    # ax.add_patch(circle)

    # next_start_point = rotate(*next_start_point, 90, circle.center)
    # square, next_start_point, _ = add_rectangle(*next_start_point, -1.0, -1.0)
    # ax.add_patch(square)

    # plt.show()

    # create_shape(state, top_level_shapes, bottom_level_shapes)
    # plt.show()
