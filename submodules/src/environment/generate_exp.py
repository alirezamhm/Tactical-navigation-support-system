import argparse
import os.path
import pickle
import random
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import packcircles as pc
from matplotlib import patches
from skimage import draw
from tqdm import tqdm

from submodules.src.geometry.polygon import *
from submodules.src.geometry.utils import Rxy
from submodules.src.utils.plot import (
    THICKNESS_CMAP,
    SHIP_PATCH_COLOR,
    OPEN_WATER_COLOR,
    CURRENTS_CMAP,
)
from submodules.src.utils.sim_utils import ICE_THICKNESS_DEFAULT, CURRENT_MAX, WavyPolynomial

SHOW_PLOT = False  # show plots of generated experiments
FLOE_SIZE_DIST = 'power'  # distribution of floe sizes, can be 'normal' or 'uniform'
EGG_CODE = {
    0.5: [  # total ice concentration
        [
            (0.1, 1),  # thickness lower bound, upper bound
            (6, 100, 10, 3),  # radius lower bound, upper bound, mean, alpha
        ],
        [
            (0.1, .5),
            (1.5, 6, 3, 2),
        ],
    ],
    0.4: [
        [
            (0.1, 1),  # thickness lower bound, upper bound
            (6, 100, 10, 3),  # radius lower bound, upper bound, mean, alpha
        ],
        [
            (0.1, .5),
            (1.5, 6, 3, 2),
        ],
    ],
    0.3: [
        [
            (0.1, 1),  # thickness lower bound, upper bound
            (6, 100, 10, 3),  # radius lower bound, upper bound, mean, alpha
        ],
        [
            (0.1, .5),
            (1.5, 6, 3, 2),
        ],
    ],
    0.2: [
        [
            (0.1, 1),  # thickness lower bound, upper bound
            (6, 100, 10, 3),  # radius lower bound, upper bound, mean, alpha
        ],
        [
            (0.1, .5),
            (1.5, 6, 3, 2),
        ],
    ],
}
START_ICE_FIELD_DIST = 100  # distance from ship starting position to start of ice field
MAP_SHAPE = (
    1400 + START_ICE_FIELD_DIST,
    500,
)  # size of rectangular ice field -- length x width in metres
SHIP_STATE = {
    # starting ship x, y position and psi heading
    "x": MAP_SHAPE[1] / 2,
    "y": 0,  # ship starts at bottom of map
    "psi": np.pi / 2,
}
GOAL = (MAP_SHAPE[1] / 2, MAP_SHAPE[0])
OBSTACLE = {
    "min_area": 3,  # min area of obstacles
    "min_y": START_ICE_FIELD_DIST,  # boundaries of where obstacles can be placed
    "max_y": MAP_SHAPE[0],
    "circular": False,  # if True, obstacles are more circular
    "num_vertices_range": (
        20,
        30,
    ),  # range for number of vertices for random convex polygons
    "final_vertices_range": (
        5,
        20,
    ),  # downsamples the polygon vertices to a number in this range
}
TOL = 0.01  # tolerance of deviation of actual concentration from desired
SCALE = 1  # scales map by this factor for occupancy grid resolution
IM_SHAPE = (int(MAP_SHAPE[0] * SCALE), int(MAP_SHAPE[1] * SCALE))

MAP_AREA = MAP_SHAPE[0] * MAP_SHAPE[1]  # area of map in m^2

SHIP_VERTICES = np.asarray(
    [
        [1.0, -0.0],
        [0.9, 0.10],
        [0.5, 0.25],
        [-1.0, 0.25],
        [-1.0, -0.25],
        [0.5, -0.25],
        [0.9, -0.10],
    ]
)

# ice parameters for ice floe generation
ICE_INIT_NUMBER_SCALES = [5, 5]  # manual scale factor for number of initial ice floes to get close to concentration

ICE_DENSITY = 900  # kg/m^3

CURRENT = True  # if True, generate currents
CURRENT_VALUE_MEAN = 0.4
CURRENT_VALUE_STD = 0.1

def compute_poly_ob_concentration(polys):
    im = np.zeros(IM_SHAPE)
    area = 0
    for p in polys:
        area += p["area"]
        rr, cc = p["pixels"]
        im[rr, cc] = 1

    return area / (MAP_SHAPE[1] * (OBSTACLE["max_y"] - OBSTACLE["min_y"])), im


def increase_concentration(obstacles, desired_concentration, partials_egg):
    actual_concentration, im = compute_poly_ob_concentration(obstacles)
    thickness_ranges = [np.array(p[0]) for p in partials_egg]
    radius_ranges = [np.array(p[1][:2]) for p in partials_egg]
    num_added = 0
    iters = 0

    while actual_concentration < desired_concentration - TOL and iters < 10000:
        if iters % 10 == 0:
            print(
                "num iters",
                iters,
                "current concentration",
                actual_concentration,
                "added obs",
                num_added,
            )
        iters += 1

        partial_choice = np.random.choice(len(radius_ranges))
        r = sample_ice_radii_from_uniform(*radius_ranges[partial_choice], size=1)[0]
        thickness = sample_thickness(*thickness_ranges[partial_choice], size=1)[0]
        slice_shape = int(max(1 / SCALE, r * 2) * SCALE)  # in map units

        # find all the slices that would fit a new obstacle using a sliding window approach
        new_obs_centres = []
        rand_offset_x = np.random.choice(np.arange(slice_shape))
        rand_offset_y = np.random.choice(np.arange(slice_shape))
        for i in range(rand_offset_y, im.shape[0] - slice_shape + 1, slice_shape):
            for j in range(rand_offset_x, im.shape[1] - slice_shape + 1, slice_shape):
                # skip if slice is beyond ice edge
                if (
                    OBSTACLE["min_y"] * SCALE <= i
                    and i + slice_shape <= OBSTACLE["max_y"] * SCALE
                ):
                    if im[i : i + slice_shape, j : j + slice_shape].sum() == 0:
                        new_obs_centres.append(
                            [
                                (j + slice_shape / 2) / SCALE,
                                (i + slice_shape / 2) / SCALE,
                            ]
                        )

        if new_obs_centres:
            # randomly choose these slices and generate an obstacle
            ind = np.random.choice(len(new_obs_centres))
            slice_choice = new_obs_centres[ind]
            x, y = slice_choice
            r = slice_shape / SCALE / 2
            vertices = generate_polygon(
                diameter=r * 2,
                origin=(x, y),
                circular=OBSTACLE["circular"],
                num_vertices_range=OBSTACLE["num_vertices_range"],
                final_vertices_range=OBSTACLE["final_vertices_range"],
            )

            if vertices is not None:
                # add ob to obstacles list
                area = poly_area(vertices)
                rr, cc = draw.polygon(
                    vertices[:, 1] * SCALE, vertices[:, 0] * SCALE, shape=IM_SHAPE
                )

                if len(rr) == 0 or im[rr, cc].sum() > 0 or area < OBSTACLE["min_area"]:
                    continue  # skip this polygon!

                obstacles.append(
                    {
                        "vertices": vertices,
                        "centre": (x, y),
                        "radius": poly_radius(vertices, centre_pos=(x, y)),
                        "pixels": (rr, cc),
                        "area": area,
                        "mass": area_to_mass(area, thickness),
                        "thickness": thickness,
                    }
                )
                num_added += 1

                # compute new poly concentration
                actual_concentration, im = compute_poly_ob_concentration(obstacles)

    print(
        "added {} obstacles over {} iterations!\ndesired concentration {}, actual concentration {}".format(
            num_added, iters, desired_concentration, actual_concentration
        )
    )

    return obstacles


def decrease_concentration(obstacles: List[dict], desired_concentration: float, n_remove=10):
    # randomize the order of obstacles
    np.random.shuffle(obstacles)
    actual_concentration, _ = compute_poly_ob_concentration(obstacles)
    num_deleted = 0
    areas = np.array([ob["area"] for ob in obstacles])
    weights = 1.0 / areas
    weights = weights / weights.sum()  # remove obstalces with probability proportional to inverse area

    while actual_concentration > desired_concentration + TOL and len(obstacles) > n_remove:
        inds = np.random.choice(np.arange(len(obstacles)), size=n_remove, replace=False, p=weights)

        obstacles = np.delete(obstacles, inds, axis=0)
        areas = np.delete(areas, inds)
        actual_concentration, _ = compute_poly_ob_concentration(obstacles)
        num_deleted += len(inds)
        if len(areas) == 0:
            break
        weights = 1.0 / areas
        weights = weights / weights.sum()

    print(
        "deleted {} obstacles!\ndesired concentration {}, actual concentration {}".format(
            num_deleted, desired_concentration, actual_concentration
        )
    )
    return obstacles


def pack_circles_and_ice_field_plot(circs, polys, pose, concentration):
    fig, ax = plt.subplots(1, 3)
    # plot circles
    for x, y, r in circs:
        patch = plt.Circle((x, y), r, fc="b", ec="k", alpha=0.65)
        ax[0].add_patch(patch)
    ax[0].set_aspect("equal")

    # compute actual concentration
    ax[0].set_title("Circle Packing Result")

    # plot polygons
    for p in polys:
        patch = patches.Polygon(
            p["vertices"], True, fill=True, fc="b", ec="k", alpha=0.65
        )
        ax[1].add_patch(patch)
    ax[1].set_aspect("equal")

    # compute actual concentration
    poly_conc, im = compute_poly_ob_concentration(polys)
    ax[1].set_title(
        "Desired Concentration {:.2f}\nActual Concentration {:.2f}\nObstacle count {}".format(
            concentration, poly_conc, len(polys)
        )
    )

    ax[2].set_title("Occupancy Grid")
    ax[2].imshow(
        im, origin="lower", cmap="gray", extent=[0, MAP_SHAPE[1], 0, MAP_SHAPE[0]]
    )

    # show ship footprint
    R = np.asarray(
        [[np.cos(pose[2]), -np.sin(pose[2])], [np.sin(pose[2]), np.cos(pose[2])]]
    )
    # show pose on plots
    for a in ax:
        a.add_patch(
            patches.Polygon(
                SHIP_VERTICES @ R.T + [pose[0], pose[1]],
                True,
                fill=True,
                fc="w",
                ec="k",
            )
        )
        a.plot(pose[0], pose[1], "rx")

    # show goal
    for a in ax:
        a.plot([0, MAP_SHAPE[1]], [GOAL[1], GOAL[1]], "g-")

    plt.show()

    floe_areas = [p["area"] for p in polys]
    floe_widths = np.sqrt(floe_areas)
    compute_fractional_area_distribution(
        floe_widths, floe_areas, MAP_SHAPE[0] * MAP_SHAPE[1]
    )

    print_ob_stats(polys)

def select_partial_conc(conc):
    possible_values = [i * 0.05 for i in range(2, int(conc / 0.05)-1)]
    conc_p1 = random.choice(possible_values)
    conc_p2 = conc - conc_p1
    return conc_p1, conc_p2

def generate_rand_exp(concs: List[float], file_name: str, trial_range: tuple):
    # dict to store the experiments
    exp_dict = {
        "meta_data": {
            "trial_range": trial_range,
            "code": EGG_CODE,
            "map_shape": MAP_SHAPE,
            "obstacle_config": OBSTACLE,
            "ship_state_config": SHIP_STATE,
            "goal": GOAL,
            "scale": SCALE,
        },
        "exp": {
            c: {
                i: {
                    "goal": None,
                    "ship_state": None,
                    "obstacles": None,
                    "currents": None,
                    "partials": None,
                }
                for i in range(*trial_range)
            }
            for c in concs
        },
    }

    # approach is to first pack the environment with circles then convert circles
    # to polygons and then do rejection sampling to attain the desired concentration
    for conc in tqdm(concs):
        for i in tqdm(range(*trial_range)):
            thicknesses = []
            radii = []
            conc_ps = select_partial_conc(conc)

            for j, conc_p in enumerate(conc_ps):

                thickness_range = EGG_CODE[conc][j][0]
                radius_dist = EGG_CODE[conc][j][1]
                avg_r = radius_dist[2]  # average radius for this partial concentration
                num_circ = int(
                    (conc_p / conc)
                    * MAP_AREA
                    / (np.pi * avg_r**2)
                    * ICE_INIT_NUMBER_SCALES[j]
                )
                thicknesses.append(sample_thickness(*thickness_range, size=num_circ))
                if FLOE_SIZE_DIST == 'uniform':
                    radii.append(sample_ice_radii_from_uniform(*radius_dist, size=num_circ))
                elif FLOE_SIZE_DIST == 'normal':
                    radii.append(sample_ice_radii_from_normal(*radius_dist, size=num_circ))
                elif FLOE_SIZE_DIST == 'power':
                    radii.append(sample_ice_radii_from_power_law(low=radius_dist[0], high=radius_dist[1], alpha=radius_dist[3], size=num_circ))

            radii = np.concatenate(radii)
            thicknesses = np.concatenate(thicknesses)

            p = np.random.permutation(len(radii))
            radii = radii[p]
            thicknesses = thicknesses[p]

            gen = pc.pack(radii)

            circles = np.asarray([(x, y, r) for (x, y, r) in gen])
            circles[:, 1] += MAP_SHAPE[0] / 2
            circles[:, 0] += MAP_SHAPE[1] / 2

            valid_indices = np.logical_and(
                circles[:, 0] >= 0, circles[:, 0] <= MAP_SHAPE[1]
            )
            valid_indices = np.logical_and(
                valid_indices,
                np.logical_and(circles[:, 1] >= 0, circles[:, 1] <= MAP_SHAPE[0]),
            )
            valid_indices = np.logical_and(
                valid_indices,
                np.logical_and(
                    circles[:, 1] >= OBSTACLE.get("min_y", 0),
                    circles[:, 1] <= OBSTACLE.get("max_y", MAP_SHAPE[0]),
                ),
            )

            circles = circles[valid_indices]
            thicknesses = thicknesses[valid_indices]

            # now generate polygons for each circle
            obstacles = []
            im = np.zeros(IM_SHAPE)

            # order the circles from largest to smallest
            # this step ensure we place the large obstacles first
            sorted_indices = np.argsort(circles[:, 2])[::-1]
            circles = circles[sorted_indices]
            thicknesses = thicknesses[sorted_indices]

            for (x, y, radius), thickness in zip(circles, thicknesses):
                vertices = generate_polygon(
                    diameter=radius * 2,
                    origin=(x, y),
                    circular=OBSTACLE["circular"],
                    num_vertices_range=OBSTACLE["num_vertices_range"],
                    final_vertices_range=OBSTACLE["final_vertices_range"],
                )

                if vertices is not None:
                    # take intersection of vertices and environment boundaries
                    vertices[:, 0][vertices[:, 0] < 0] = 0
                    vertices[:, 0][vertices[:, 0] >= MAP_SHAPE[1]] = MAP_SHAPE[1]

                    min_y = OBSTACLE.get("min_y", 0)
                    max_y = OBSTACLE.get("max_y", MAP_SHAPE[0])
                    vertices[:, 1][vertices[:, 1] < min_y] = min_y
                    vertices[:, 1][vertices[:, 1] > max_y] = max_y

                    area = poly_area(vertices)
                    rr, cc = draw.polygon(
                        vertices[:, 1] * SCALE, vertices[:, 0] * SCALE, shape=IM_SHAPE
                    )

                    if (
                        len(rr) == 0
                        or im[rr, cc].sum() > 0
                        or area < OBSTACLE["min_area"]
                    ):
                        # skip this polygon if:
                        #  - it doesn't fit in the environment
                        #  - it overlaps with another obstacle
                        #  - it's area is too small
                        continue

                    obstacles.append(
                        {
                            "vertices": vertices,
                            "centre": (x, y),
                            "radius": poly_radius(vertices, centre_pos=(x, y)),  # m
                            "pixels": (rr, cc),
                            "area": area,  # area in m^2
                            "mass": area_to_mass(area, thickness),  # mass in kg
                            "thickness": thickness,  # thickness in m
                        }
                    )

                    # compute new poly concentration
                    actual_concentration, im = compute_poly_ob_concentration(obstacles)

            # get concentration of ice field with polygon obstacles
            poly_concentration, _ = compute_poly_ob_concentration(obstacles)
            if abs(conc - poly_concentration) > TOL:
                print(
                    "\ndesired concentration {}, actual concentration {}".format(
                        conc, poly_concentration
                    )
                )
                if conc > poly_concentration:
                    # randomly add obstacles:
                    obstacles = increase_concentration(obstacles, conc, EGG_CODE[conc])
                else:
                    obstacles = decrease_concentration(obstacles, conc)

            exp_dict["exp"][conc][i]["obstacles"] = obstacles

            exp_dict["exp"][conc][i]["goal"] = GOAL

            ship_state = (SHIP_STATE["x"], SHIP_STATE["y"], SHIP_STATE["psi"])
            exp_dict["exp"][conc][i]["ship_state"] = ship_state

            currents = generate_currents(MAP_SHAPE[1], MAP_SHAPE[0]) if CURRENT else None
            exp_dict["exp"][conc][i]["currents"] = currents
            
            exp_dict["exp"][conc][i]["partials"] = conc_ps

            if SHOW_PLOT:
                pack_circles_and_ice_field_plot(circles, obstacles, ship_state, conc)

    if file_name:
        with open(file_name, "wb") as f:
            pickle.dump(exp_dict, f)
            print("Saved experiment configuration file to", os.path.abspath(file_name))
    print("Done!")


def build_obs_dicts(obstacles: List):
    obs_dicts = []
    for p in obstacles:
        p = np.asarray(p)
        centre = poly_centroid(p)
        area = poly_area(p)
        obs_dicts.append(
            {
                "vertices": p,
                "centre": centre,
                "radius": poly_radius(p, centre),
                "area": area,
                "mass": area_to_mass(area),
            }
        )
    return obs_dicts


def sample_ice_radii_from_uniform(low: float, high: float, size=1):
    """
    Samples ice radii from a uniform distribution.
    This function generates random radii values sampled from a uniform
    distribution within the specified range [low, high].
    Parameters:
    -----------
    low : float, optional
        The lower bound of the uniform distribution. Default is 0.1.
    high : float, optional
        The upper bound of the uniform distribution. Default is 2.
    size : int, optional
        The number of samples to generate. Default is 1.
    Returns:
    --------
    numpy.ndarray
        An array of sampled radii values.
    """
    radii = np.random.uniform(low=low, high=high, size=size)
    return radii

def sample_ice_radii_from_normal(low: float, high: float, mean: float, std: float, size=1):
    """
    Samples ice radii from a normal distribution and clips the values to [low, high].
    Parameters:
    -----------
    low : float
        The lower bound for the radii.
    high : float
        The upper bound for the radii.
    mean : float
        The mean of the normal distribution.
    std : float
        The standard deviation of the normal distribution.
    size : int, optional
        The number of samples to generate. Default is 1.
    Returns:
    --------
    numpy.ndarray
        An array of sampled and clipped radii values.
    """
    radii = np.random.normal(loc=mean, scale=std, size=size)
    radii = np.clip(radii, low, high)
    return radii

def sample_ice_radii_from_power_law(low: float, high: float, alpha: float, size=1):
    """
    Samples ice radii from a power law distribution.
    This function generates random radii values sampled from a power law distribution
    within the specified range [low, high].
    Parameters:
    -----------
    low : float
        The lower bound of the distribution.
    high : float
        The upper bound of the distribution.
    alpha : float
        The exponent of the power law distribution.
    size : int, optional
        The number of samples to generate. Default is 1.
    Returns:
    --------
    numpy.ndarray
        An array of sampled radii values.
    """
    u = np.random.uniform(size=size)
    radii = ((high**(1-alpha) - low**(1-alpha)) * u + low**(1-alpha)) ** (1 / (1-alpha))
    radii = np.clip(radii, low, high)
    return radii


def compute_fractional_area_distribution(widths, areas, total_area, bin_spacing=100):
    widths = np.asarray(widths)
    areas = np.asarray(areas)

    min_width = np.min(widths)
    max_width = np.max(widths)
    bins = np.linspace(min_width, max_width, bin_spacing)

    frac_area = []
    for p in bins:
        frac_area.append(np.sum(areas[widths > p]) / total_area)

    plt.plot(bins, frac_area)
    plt.xlabel("Effective floe width")  # aka characteristic Length
    plt.ylabel("Fractional area")
    plt.xscale("log")
    plt.show()


def plot_cdf(masses, bin_spacing=100):
    min_mass = np.min(masses)
    max_mass = np.max(masses)
    bins = np.linspace(min_mass, max_mass, bin_spacing)

    cdf = []
    for p in bins:
        cdf.append(sum(masses <= p) / len(masses))

    plt.plot(bins, cdf)
    plt.xlabel("Floe mass (kg)")
    plt.ylabel("Cumulative probability")
    plt.xscale("log")
    plt.show()


def print_ob_stats(obs):
    print("number of obstacles", len(obs))
    print("mean area", np.mean([p["area"] for p in obs]))
    print("std area", np.std([p["area"] for p in obs]))
    print("min area", np.min([p["area"] for p in obs]))
    print("max area", np.max([p["area"] for p in obs]))
    print("mean characteristic length", np.mean([np.sqrt(p["area"]) for p in obs]))
    print("std characteristic length", np.std([np.sqrt(p["area"]) for p in obs]))
    print("min characteristic length", np.min([np.sqrt(p["area"]) for p in obs]))
    print("max characteristic length", np.max([np.sqrt(p["area"]) for p in obs]))
    if "thickness" in obs[0]:
        print("mean thickness", np.mean([p["thickness"] for p in obs]))
        print("std thickness", np.std([p["thickness"] for p in obs]))


def view_experiments(pickle_file, num_trials_per_plot=10, num_trials_skip=10):
    with open(pickle_file, "rb") as f:
        exp_dict = pickle.load(f)

    curr_plot_count = 0

    for c in exp_dict["exp"]:
        print("\nConcentration:", c)
        for i in exp_dict["exp"][c]:
            if num_trials_skip is not None and i % num_trials_skip != 0:
                continue
            obs = exp_dict["exp"][c][i]["obstacles"]
            print(f"\nTrial {i}")
            print("obs count", len(obs))
            ship_state = exp_dict["exp"][c][i]["ship_state"]
            map_shape = exp_dict["meta_data"]["map_shape"]

            currents = exp_dict["exp"][c][i].get("currents", None)
            conc_ps = exp_dict["exp"][c][i].get("partials", None)

            ice_field_plot(c, i, ship_state, obs, currents, map_shape, conc_ps)
            curr_plot_count += 1

            print_ob_stats(obs)

            if curr_plot_count == num_trials_per_plot:
                curr_plot_count = 0
                plt.show()

    all_obs = []
    for c in exp_dict["exp"]:
        for i in exp_dict["exp"][c]:
            all_obs.extend(exp_dict["exp"][c][i]["obstacles"])
    print("\nall obs")
    print_ob_stats(all_obs)


def ice_field_plot(
    concentration,
    ice_field_idx,
    ship_state,
    obs,
    currents=None,
    map_shape=None,
    conc_ps=None,
    ship_vertices=None,
):
    fig, ax = plt.subplots(figsize=(10, 10))

    min_thickness = min(p["thickness"] for p in obs)
    max_thickness = max(p["thickness"] for p in obs)
    sm_thickness = plt.cm.ScalarMappable(
        cmap=THICKNESS_CMAP, norm=plt.Normalize(vmin=min_thickness, vmax=max_thickness)
    )

    cbar_thickness = plt.colorbar(sm_thickness, ax=ax, orientation="vertical", pad=0.02)
    cbar_thickness.set_label("Ice Thickness (m)")

    for p in obs:
        color = sm_thickness.to_rgba(p["thickness"])

        patch = patches.Polygon(
            p["vertices"], True, fill=True, fc=color, ec="k", linewidth=0.5, zorder=0
        )
        ax.add_patch(patch)

    if currents:
        sm_currents = plt.cm.ScalarMappable(
            cmap=CURRENTS_CMAP, norm=plt.Normalize(vmin=-CURRENT_MAX, vmax=CURRENT_MAX)
        )
        sm_currents.set_array([])  # Only needed for the colorbar
        
        # Sample y-coordinates along the map height
        y_samples = np.linspace(currents.y_min, currents.y_max, 500)
        x_samples = currents(y_samples)
        
        current_width = currents.width
        current_speed = currents.speed
        print("Current speed:", current_speed)
        color = sm_currents.to_rgba(current_speed)
        
        x_left = x_samples - current_width / 2
        x_right = x_samples + current_width / 2
        
        # Create vertices for the filled polygon (left side going up, right side going down)
        vertices = np.vstack([
            np.column_stack([x_left, y_samples]),
            np.column_stack([x_right[::-1], y_samples[::-1]])
        ])
        
        # Add the current region as a filled polygon
        current_patch = patches.Polygon(
            vertices,
            closed=True,
            facecolor=color,
            edgecolor='none',
            zorder=-1
        )
        ax.add_patch(current_patch)

        
        # Add colorbar for currents
        cbar_currents = plt.colorbar(
            sm_currents, ax=ax, orientation="vertical", pad=0.02
        )
        cbar_currents.set_label("Current Value")

    ax.set_aspect("equal")
    ax.plot(ship_state[0], ship_state[1], "rx")
    ax.set_title(
        f"Concentration: {concentration:.2f} ({conc_ps[0]:.2f}, {conc_ps[1]:.2f}) Ice Field Index: {ice_field_idx}\nObs count: {len(obs)}"
    )

    if map_shape is not None:
        ax.set_xlim(0, map_shape[1])
        ax.set_ylim(0, map_shape[0] + 4)

    if ship_vertices is not None:
        ax.add_patch(
            patches.Polygon(
                ship_vertices @ Rxy(ship_state[2]).T + [ship_state[0], ship_state[1]],
                True,
                fill=True,
                fc=SHIP_PATCH_COLOR,
                ec="k",
            )
        )
    ax.set_facecolor(OPEN_WATER_COLOR)


def mass_to_area(ice_mass, ice_thickness=ICE_THICKNESS_DEFAULT):
    """
    Calculate the area of ice given its mass and thickness.
    Parameters:
    ice_mass (float or array-like): The mass of the ice in kilograms.
    ice_thickness (float or array-like, optional): The thickness of the ice in meters.
                                                   Default is ICE_THICKNESS.
                                                   Must be an int or array with the same size as ice_mass.
    Returns:
    float or array-like: The area of the ice in square meters.
    """

    return ice_mass / (ICE_DENSITY * ice_thickness)


def area_to_mass(ice_area, ice_thickness=ICE_THICKNESS_DEFAULT):
    """
    Calculate the mass of ice given its area and thickness.
    Parameters:
    ice_area (float or array-like): The area of the ice in square meters.
    ice_thickness (float or array-like, optional): The thickness of the ice in meters.
                                                   Default is ICE_THICKNESS.
                                                   Must be an int or array with the same size as ice_area.
    Returns:
    float or array-like: The mass of the ice in kilograms.
    """
    return ice_area * ICE_DENSITY * ice_thickness


def area_to_radii(ice_area):
    return np.sqrt(ice_area / np.pi)


def sample_thickness(low=0.1, high=2, size=1):
    """
    Generate a random uniform sample of ice thickness.
    This function returns an array of random samples from a unifrom distribution between low and high.
    Args:
        low (float): The lower bound of the ice thickness.
        high (float): The upper bound of the ice thickness.
        size (int): The number of samples to generate.
    Returns:
        np.ndarray: An array of random samples representing ice thickness.
    """
    return np.random.uniform(low=low, high=high, size=size)

def generate_currents(map_width: int, map_height: int, no_current_p: float = 0.2) -> WavyPolynomial:
    """
    Generate a wavy polynomial current using the WavyPolynomial class.
    
    This function creates a wavy current pattern across the map. The current has a random
    width (10-30% of map width), is positioned randomly along the x-axis, and has a 
    y-extent covering the entire map height. The current speed is sampled from a normal
    distribution and used as a scaling factor for the wave amplitude.
    
    Args:
        map_width (int): The width of the map in meters.
        map_height (int): The height of the map in meters.
        no_current_p (float): Probability of generating no current (default is 0.2).
    
    Returns:
        WavyPolynomial | None: A WavyPolynomial object representing the current pattern,
                               or None if no current is generated.
    """
    # Random chance of no current
    if np.random.uniform() < no_current_p:
        return None
    
    # Sample current speed from normal distribution
    current_speed = np.random.normal(loc=CURRENT_VALUE_MEAN, scale=CURRENT_VALUE_STD)
    if np.random.uniform() < 0.5:
        current_speed *= -1  # Randomly reverse direction
    current_speed = np.clip(current_speed, -CURRENT_MAX, CURRENT_MAX)
    
    y_min = 0
    y_max = map_height
    
    x_min = 0
    x_max = map_width
    
    # Generate random polynomial coefficients
    degree = 10
    coeffs = np.random.normal(0, 1, degree)
    coeffs = coeffs / (np.arange(1, degree + 1))

    
    # Generate random current width
    current_width = np.random.uniform(0.1, 0.3) * map_width
    
    # Create and return WavyPolynomial
    wavy_current = WavyPolynomial(
        coeffs=coeffs,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        width=current_width,
        speed=current_speed
    )
    
    return wavy_current


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random ice field experiments.")
    parser.add_argument("--conc", nargs="+", type=int, choices=[20, 30, 40, 50], required=True, help="List of concentrations to generate, values must be from [20, 30, 40, 50].")
    parser.add_argument("-f", "--file_name", type=str, required=True, help="File name to save the experiments.")
    parser.add_argument("--trial_range", type=int, nargs=2, default=(0, 5), help="Range of trials to generate for each concentration [start, end).")
    args = parser.parse_args()
    
    concs = [c / 100 for c in args.conc]
    
    generate_rand_exp(concs, file_name=args.file_name, trial_range=args.trial_range)
    # view_experiments(args.file_name, num_trials_per_plot=5, num_trials_skip=None)
