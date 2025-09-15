import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_vibrant_vectors(vectors_set1, vectors_set2=None, figsize=(5, 5), arrow_scale=1.0, line_width=6):
    """
    Plot two sets of 2-dimensional vectors with different styles.
    
    Parameters:
    -----------
    vectors_set1 : array-like, shape (n, 2)
        First set of vectors - shown as bright yellow dotted lines extending to edges
    vectors_set2 : array-like, shape (m, 2), optional
        Second set of vectors - shown in rainbow colors on top of first set
    figsize : tuple, default (5, 5)
        Figure size in inches
    arrow_scale : float, default 1.0
        Scale factor for arrow length
    line_width : float, default 5
        Width of the arrow lines
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    
    # Convert to numpy arrays if needed
    vectors_set1 = np.array(vectors_set1) if vectors_set1 is not None else np.array([])
    vectors_set2 = np.array(vectors_set2) if vectors_set2 is not None else np.array([])
    
    def generate_vibrant_colors(n):
        """Generate vibrant colors using HSV color space"""
        colors = []
        for i in range(n):
            # Use full hue range with high saturation and value for vibrant colors
            hue = i / n  # Evenly space hues around the color wheel
            saturation = 0.9 + 0.1 * (i % 2)  # Very high saturation (0.9-1.0)
            value = 0.8 + 0.2 * ((i // 2) % 2)  # High brightness (0.8-1.0)
            rgb = mcolors.hsv_to_rgb([hue, saturation, value])
            colors.append(rgb)
        return colors
    
    def extend_line_to_boundary(start, direction, xlim, ylim):
        """Extend a line from start point in given direction to plot boundary"""
        # Normalize direction
        if np.linalg.norm(direction) == 0:
            return start, start
        
        direction = direction / np.linalg.norm(direction)
        
        # Calculate intersections with all four boundaries
        intersections = []
        
        # Right boundary (x = xlim[1])
        if direction[0] > 0:
            t = (xlim[1] - start[0]) / direction[0]
            y = start[1] + t * direction[1]
            if ylim[0] <= y <= ylim[1]:
                intersections.append([xlim[1], y])
        
        # Left boundary (x = xlim[0])
        if direction[0] < 0:
            t = (xlim[0] - start[0]) / direction[0]
            y = start[1] + t * direction[1]
            if ylim[0] <= y <= ylim[1]:
                intersections.append([xlim[0], y])
        
        # Top boundary (y = ylim[1])
        if direction[1] > 0:
            t = (ylim[1] - start[1]) / direction[1]
            x = start[0] + t * direction[0]
            if xlim[0] <= x <= xlim[1]:
                intersections.append([x, ylim[1]])
        
        # Bottom boundary (y = ylim[0])
        if direction[1] < 0:
            t = (ylim[0] - start[1]) / direction[1]
            x = start[0] + t * direction[0]
            if xlim[0] <= x <= xlim[1]:
                intersections.append([x, ylim[0]])
        
        # Return the closest intersection
        if intersections:
            distances = [np.linalg.norm(np.array(point) - start) for point in intersections]
            closest_idx = np.argmin(distances)
            return start, intersections[closest_idx]
        else:
            return start, start + direction * 0.1  # Fallback
    
    # Create the plot with white background
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    
    # Set the axis limits first
    xlim = (-1.2, 1.2)
    ylim = (-1.2, 1.2)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Plot first set of vectors (bright yellow, dotted, extending to edges)
    if len(vectors_set1) > 0:
        bright_yellow = '#FFD700'  # Bright yellow color
        
        # Sort by norm for consistent layering
        vector_norms = np.linalg.norm(vectors_set1, axis=1)
        sorted_indices = np.argsort(vector_norms)[::-1]  # Descending order
        
        for idx in sorted_indices:
            vector = vectors_set1[idx]
            scaled_vector = vector * arrow_scale
            
            # Extend line to boundary
            start_point, end_point = extend_line_to_boundary([0, 0], scaled_vector, xlim, ylim)
            
            # Plot dotted line from origin to boundary
            ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                    color=bright_yellow, linewidth=line_width * 1.5, alpha=0.9, 
                    linestyle='--', zorder=1)
            
            # Plot dot at the scaled vector endpoint
            ax.scatter(scaled_vector[0], scaled_vector[1], 
                      s=350, c=bright_yellow, marker='o', 
                      edgecolors='white', linewidth=2,
                      alpha=1, zorder=2)
    
    # Plot second set of vectors (hard-coded colors, on top)
    if len(vectors_set2) > 0:
        fixed_colors = ['#d62a85', '#0fc5e6', '#b5ef08']
        
        for i, vector in enumerate(vectors_set2):
            if i >= 3:
                break
            scaled_vector = vector * arrow_scale
            
            # Plot line from origin to vector endpoint
            ax.plot([0, scaled_vector[0]], [0, scaled_vector[1]], 
                    color=fixed_colors[i], linewidth=line_width * 0.85, alpha=1, zorder=3)
            
            # Plot dot at the end of the vector
            ax.scatter(scaled_vector[0], scaled_vector[1], 
                      s=180, c=[fixed_colors[i]], marker='o', 
                      edgecolors='white', linewidth=0.2,
                      alpha=1, zorder=4)
    
    # Make axes equal and add grid
    ax.set_aspect('equal', adjustable='box')
    
    # Add axis lines through origin
    ax.axhline(y=0, color='gray', linewidth=6, alpha=0.3)
    ax.axvline(x=0, color='gray', linewidth=6, alpha=0.3)
    
    # Remove ticks and labels for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Remove spines for minimal look
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    
    return fig, ax