#!/usr/bin/env python
"""Generate figures showing the min-level set of a learned
mapping, its pre-image under an inverse network, 
and trajectories generated from goal-condition neural ODE.

* compute the mapped boundary of the domain using a trained PLNet
* find the minimum radius of that contour around the mapped zero point
* sample points on the corresponding circle in latent/model space
* push those samples back through the inverse network
* draw one figure in model space and another in original (input) space
* goal-conditioned neural ODE trajectories from the sampled points to the 
    varying zero points in both spaces

All extraneous code has been stripped; only the essentials are kept.

Author: Dechuan Liu (April 2026)
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
import matplotlib.pyplot as plt
from flax import linen as nn 
from robustnn.plnet_jax import PLNet
from robustnn.bilipnet_jax import BiLipNet
from utils import normalize_data, generate_path_in_gmap_space, monotone_uniform_map, draw_boundary, plot_gradient_descent_path, plot_value_contour, sample_points_on_line, get_line_and_value, sample_linear_on_sphere_boundary, sample_linear_grid_points

def main():
    is_plotting_circle = True
    is_plotting_maze_figure = True
    is_plotting_path_fixed = True
    is_plotting_path_varying_z = True

    # configuration -------------------------------------------------------
    x_range = [-8, 8]
    normalized_range = [0, 1]
    zero_point = (0.5, 7 / 8)
    rng = jax.random.PRNGKey(42)

    # network hyper‑parameters (must match the saved model) --------------
    data_dim = 2
    layer_size = [128] * 4
    depth = 12
    mu = 0.1
    nu = 128
    prefix = os.getcwd() + '/results/2D-corridor/figures/'
    os.makedirs(os.path.dirname(prefix), exist_ok=True)

    # path to the trained parameters (adjust if you use a different run) -
    train_dir_pl = os.getcwd() + '/results/2D-corridor/model/pl'

    # instantiate model --------------------------------------------------
    model_bilip = BiLipNet(input_size=data_dim, 
                     units=layer_size, 
                     depth=depth, 
                     mu=mu, 
                     nu=nu)
    model_pl = PLNet(model_bilip, optimal_point=jnp.array(zero_point))

    # load checkpoint -----------------------------------------------------
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    params_pl = orbax_checkpointer.restore(f'{train_dir_pl}/ckpt/params')
    params_bilip = {'params':params_pl['params']['BiLipBlock']}
    params_bilip_inv = model_bilip.direct_to_explicit_inverse(params_bilip,
                                                        [1.0]*depth,
                                                        [nn.relu]*depth,
                                                        [500]*depth,
                                                        [1.0]*depth)
    params_bilip_explicit = model_bilip.direct_to_explicit(params_bilip)

    def network_gmapping_pl(point):
        return model_bilip.explicit_call(params_bilip, point, params_bilip_explicit)

    # --- compute value function on a regular grid for contour plotting ---
    sample_in_axis = 200
    sample = sample_linear_grid_points(normalized_range, normalized_range,
                                       sample_in_axis, sample_in_axis)
    def network_output_pl(point):
        return model_pl.apply(params_pl, point, jnp.array(zero_point))
    
    def sorted_network_output_pl(point):
        # sort the output
        output = model_pl.apply(params_pl, point, jnp.array(zero_point))
        output = monotone_uniform_map(output, output_range=(0, 1))
        return output

    model_output_pl = network_output_pl(sample)

    # sample a dense set of points on the physical boundary (normalized) ----
    lines, values = get_line_and_value(inner_radius=-0.1)
    pts, _ = sample_points_on_line(lines, np.full_like(values, np.inf),
                                   n_samples=50000)
    pts = jnp.array(pts)
    sample_arrow_on_boundary = normalize_data(pts, x_range, normalized_range)

    contour_lines_mapped = network_gmapping_pl(sample_arrow_on_boundary)
    zero_in_model = network_gmapping_pl(jnp.array(zero_point))

    # determine the smallest radius from the mapped zero point ------------
    min_level = jnp.min(jnp.linalg.norm(contour_lines_mapped - zero_in_model, axis=1))
    
    # sample points on the circle (sphere in 2‑D) in latent/model space -----
    num_samples = 8000
    dim = 2
    sphere_samples = sample_linear_on_sphere_boundary(
        rng, num_samples, center=zero_in_model, radius=min_level
    )

    def inverse_func(data):
        return model_bilip.inverse_call(params_bilip, data, params_bilip_inv)

    # map everything back to input space ---------------------------------
    sphere_orig = inverse_func(sphere_samples)
    selected_boundary_z = []
    gap = np.floor(len(sphere_samples)/15) 
    for i in range(15):
        selected_boundary_z.append((sphere_samples[int((i+1)*gap)]-zero_in_model) * np.random.uniform(low=0.2, high=0.9)+zero_in_model)

    # calculate start and end
    start_points_z = jnp.array(selected_boundary_z)
    start_points = inverse_func(start_points_z)

    colors = [
              "#a6cee3",
              "#1f78b4", 
              "#b2df8a",
              "#33a02c", 
              "#fb9a99",
              "#fdbf6f", 
              "#afcc0a",
              "#cab2d6", 
              "#6a3d9a",
              "#543005",
              "#000000",
              "#8c510a",
              "#003c30",
              "#01665e",
              "#4C660F",
              "#721212",
              ]  # blue, red, green, purple
    if len(colors) < len(start_points):
        exit(f"error in color space: {len(colors)} - {len(start_points)}")

    # plot directories -----------------------------------------------------
    os.makedirs('figures', exist_ok=True)


    if is_plotting_circle:
        # plot 1: model space --------------------------------------------------
        # background showing a (normalized) unit ball centered at the mapped zero point
        fig1, ax1 = plt.subplots(figsize=(4, 4))
        # --- Data setup ---
        range_ = [-6, 6]

        x = np.linspace(zero_in_model[0] + range_[0], zero_in_model[0] + range_[1], 300)
        y = np.linspace(zero_in_model[1] + range_[0], zero_in_model[1] + range_[1], 300)
        X, Y = np.meshgrid(x, y)

        # Example surface
        Z = (X - zero_in_model[0])**2 + (Y - zero_in_model[1])**2
        
        # fill and contour to resemble earlier notebook's unit‑disk view
        import matplotlib.colors as mcolors
        vmin = 0
        vmax = 42
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=False)
        contour = ax1.contourf(
                X, Y, Z, levels=np.linspace(vmin, vmax, 20), cmap='viridis', alpha=0.95, norm=norm, extend='max'
        )
        ax1.plot(sphere_samples[:, 0], sphere_samples[:, 1], 
                linewidth=2.5, label='sampled circle in model space',
                color='#d73027')

        ax1.axis('equal')
        # ax1.legend(frameon=False, fontsize=8)
        # Hide frame
        for spine in ax1.spines.values():
            spine.set_visible(False)
        ax1.set_xticks([]); ax1.set_yticks([])
        fig1.savefig(f'{prefix}model_space_boundary.pdf', dpi=300)
        plt.close(fig1)

    if is_plotting_maze_figure:
        # plot 2: original/input space ----------------------------------------
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        # optional: show contours of the latent value mapped back (for visualization)
        # here we simply reuse the same sample grid for context, though the values
        # are meaningless in input coordinates. Adjust or remove as needed.

        #  give scaling for model
        model_output_pl_sorted = sorted_network_output_pl(sample)
        plot_value_contour(
            ax2, sample, model_output_pl_sorted,
            title='Original space (reference)', vlim=[0, jnp.max(model_output_pl_sorted)],
            xlim=normalized_range, ylim=normalized_range,
        )

        draw_boundary(ax2, width=1, color='black', closed=False)
        ax2.plot(sphere_orig[:, 0], sphere_orig[:, 1], c='#d73027', linewidth=2.5,
                    label='sampled circle preimage')
        
        ax2.axis('equal')

        # ax2.legend(frameon=False, fontsize=8)
        for spine in ax2.spines.values():
            spine.set_visible(False)
        ax2.set_xticks([]); ax2.set_yticks([])
    
        # remove tile for cleaner look; adjust as needed
        ax2.set_title('', fontsize=12)
        fig2.savefig(f'{prefix}original_space.pdf', dpi=300)
        plt.close(fig2)

    if is_plotting_path_fixed:
        plt.subplots(figsize=(4, 4))
        ax = plt.gca()

        # --- Professional line styling ---
        ax.plot(
            sphere_samples[:, 0], sphere_samples[:, 1],
            color='#d73027', linewidth=2.5, alpha=0.9,
            solid_capstyle='round', label='Boundary'
        )

        # --- Add arrows from contour points to zero_in_model ---
        # Sample points to avoid overcrowding (adjust step size as needed)
        for i in range(len(start_points)):
            point = start_points_z[i]
            ax.plot(point[0], point[1], 'o', color=colors[i], markersize=6, zorder=5)
            ax.annotate('', 
                        xy=(zero_in_model[0], zero_in_model[1]),  # Arrow points to zero
                        xytext=(point[0], point[1]),  # Arrow starts at contour point
                        arrowprops=dict(
                            arrowstyle='->', 
                            color=colors[i],  # Blue color for contrast
                            lw=3, 
                            alpha=0.8,
                            shrinkA=0,  # Don't shrink at start
                            shrinkB=3   # Small shrink at end to avoid overlapping center
                        ))
            
        # Mark the zero point
        ax.plot(zero_in_model[0], zero_in_model[1], 'x', 
                color = "#ff6600", markersize=6, markeredgewidth=2, label='Zero', zorder=5)

        # --- Labeling and aesthetics ---
        ax.set_aspect('equal', 'box')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("", fontsize=14, fontname="serif", labelpad=6)
        ax.set_ylabel("", fontsize=14, fontname="serif", labelpad=6)
        # ax.legend(frameon=False, fontsize=11, loc='upper right')

        # Hide frame
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()
        plt.savefig(f'{prefix}original_maze_fixed.pdf',
                    dpi=300, bbox_inches='tight')
        
        # --------------------------------------------------------------------------------------------
        plt.subplots(figsize=(4, 4))
        ax = plt.gca()

        # --- Professional line styling ---
        draw_boundary(ax, width=1, color='black', closed=False)
        ax.plot(sphere_orig[:, 0], sphere_orig[:, 1], c='#d73027', linewidth=2.5,
                    label='sampled circle preimage')


        # --- Add arrows from contour points to zero_in_model ---
        # Sample points to avoid overcrowding (adjust step size as needed)
        points_y = generate_path_in_gmap_space(start_points, 
            network_gmapping_pl, 
            jnp.array(zero_point), 1000, step_size=0.01)

        # draw every path
        for i in range(len(points_y)):
            point_y = points_y[i]
            path = inverse_func(point_y)
            # print(path)
            ax.plot(path[0][0], path[0][1], 'o', color=colors[i], markersize=6, zorder=5)
            plot_gradient_descent_path(ax, path,
                                    title='',
                                    xlim=normalized_range, ylim=normalized_range, 
                                    path_color=colors[i], 
                                    is_drawing_arrow=True, arrowsize=4,linewidth=3)

        # Mark the zero point
        ax.plot(zero_point[0], zero_point[1], 'x', 
                color = "#ff6600", markersize=6, markeredgewidth=2, label='Zero', zorder=5)

        # --- Labeling and aesthetics ---
        ax.set_aspect('equal', 'box')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("", fontsize=14, fontname="serif", labelpad=6)
        ax.set_ylabel("", fontsize=14, fontname="serif", labelpad=6)
        # ax.legend(frameon=False, fontsize=11, loc='upper right')

        # Hide frame
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()
        plt.savefig(f'{prefix}maze_fixed.pdf',
                    dpi=300, bbox_inches='tight')
        
    if is_plotting_path_varying_z:
        
        zero_point_new = (11/16, 7/16)
        zero_in_model_new = network_gmapping_pl(jnp.array(zero_point_new))
        plt.subplots(figsize=(4, 4))
        ax = plt.gca()

        # --- Professional line styling ---
        ax.plot(
            sphere_samples[:, 0], sphere_samples[:, 1],
            color='#d73027', linewidth=2.5, alpha=0.9,
            solid_capstyle='round', label='Boundary'
        )

        # --- Add arrows from contour points to zero_in_model ---
        for i in range(len(start_points_z)):
            point = start_points_z[i]
            ax.plot(point[0], point[1], 'o', color=colors[i], markersize=6, zorder=5)
            ax.annotate('', 
                        xy=(zero_in_model_new[0], zero_in_model_new[1]),  # Arrow points to zero
                        xytext=(point[0], point[1]),  # Arrow starts at contour point
                        arrowprops=dict(
                            arrowstyle='->', 
                            color=colors[i],  # Blue color for contrast
                            lw=3, 
                            alpha=0.8,
                            shrinkA=0,  # Don't shrink at start
                            shrinkB=3   # Small shrink at end to avoid overlapping center
                        ))
            
        # Mark the zero point
        ax.plot(zero_in_model_new[0], zero_in_model_new[1], 'x', 
                color="#ff6600", markersize=6, markeredgewidth=2, label='Zero', zorder=5)

        # --- Labeling and aesthetics ---
        ax.set_aspect('equal', 'box')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("", fontsize=14, fontname="serif", labelpad=6)
        ax.set_ylabel("", fontsize=14, fontname="serif", labelpad=6)
        # ax.legend(frameon=False, fontsize=11, loc='upper right')

        # Hide frame
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()
        plt.savefig(f'{prefix}original_multi_start.pdf',
                    dpi=300, bbox_inches='tight')
        
        # --------------------------------------------------------------------------------------------
        plt.subplots(figsize=(4, 4))
        ax = plt.gca()

        # --- Professional line styling ---
        draw_boundary(ax, width=1, color='black', closed=False)
        ax.plot(sphere_orig[:, 0], sphere_orig[:, 1], c='#d73027', linewidth=2.5,
                    label='sampled circle preimage')

        # --- Add arrows from contour points to zero_in_model_new ---
        # Sample points to avoid overcrowding (adjust step size as needed)
        points_y = generate_path_in_gmap_space(start_points, 
            network_gmapping_pl, 
            jnp.array(zero_point_new), 1000, step_size=0.01)

        # draw every path
        for i in range(len(points_y)):
            point_y = points_y[i]
            path = inverse_func(point_y)
            ax.plot(path[0][0], path[0][1], 'o', color=colors[i], markersize=6, zorder=5)
            plot_gradient_descent_path(ax, path,
                                    title='',
                                    xlim=normalized_range, ylim=normalized_range, 
                                    path_color=colors[i], 
                                    is_drawing_arrow=True, arrowsize=4,linewidth=3)

        # Mark the zero point
        ax.plot(zero_point_new[0], zero_point_new[1], 'x', 
                color = "#ff6600", markersize=6, markeredgewidth=2, label='Zero', zorder=5)

        # --- Labeling and aesthetics ---
        ax.set_aspect('equal', 'box')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("", fontsize=14, fontname="serif", labelpad=6)
        ax.set_ylabel("", fontsize=14, fontname="serif", labelpad=6)
        # ax.legend(frameon=False, fontsize=11, loc='upper right')

        # Hide frame
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()
        plt.savefig(f'{prefix}maze_multi_start.pdf',
                    dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()
