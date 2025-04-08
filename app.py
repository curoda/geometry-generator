import streamlit as st
import numpy as np
import pandas as pd
import math

st.title("DAM GEOMETRY GENERATOR")

# -----------------------------------------------------------------------------
# Input parameters
# -----------------------------------------------------------------------------
# M and N are the grid dimensions (with N = M)
M = st.number_input("Enter grid size M (and N will be the same):", value=10, step=1, min_value=1)
K = st.number_input("Enter multiplier K:", value=100, step=1)
T = st.number_input("Enter number of points T (must be a multiple of 8):", value=24, step=1)

if T % 8 != 0:
    st.error("T must be a multiple of 8.")
    st.stop()

N_points = M * M  # since grid is M x M

# -----------------------------------------------------------------------------
# Fibonacci Sphere function
# -----------------------------------------------------------------------------
def fibonacci_sphere(n):
    """
    Generate n points on the surface of a unit sphere using the Fibonacci spiral method.
    Returns a list of (x, y, z) tuples.
    """
    points = []
    # When only one candidate, return the north pole.
    if n == 1:
        return [(0, 0, 1)]
    golden_angle = math.pi * (3 - math.sqrt(5))
    for i in range(n):
        # y goes from 1 to -1
        y = 1 - (i / (n - 1)) * 2  
        # radius at y level from the sphere
        radius = math.sqrt(max(0, 1 - y * y))
        theta = golden_angle * i
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append((x, y, z))
    return points

# Generate sphere points for each grid point (total M*M points)
sphere_points = fibonacci_sphere(N_points)

# -----------------------------------------------------------------------------
# Build candidate list
# -----------------------------------------------------------------------------
candidates = []
# We use the grid order to label candidates: index i corresponds to (m, n)
for idx, (x, y, z) in enumerate(sphere_points):
    m = idx // M
    n = idx % M
    # Multiply by K and round to nearest integer:
    X = round(x * K)
    Y = round(y * K)
    Z = round(z * K)
    
    # Filter condition: each coordinate must have an absolute value between 10 and 90 (inclusive)
    if not (10 <= abs(X) <= 90 and 10 <= abs(Y) <= 90 and 10 <= abs(Z) <= 90):
        continue

    # Determine octant for (X, Y, Z); note that 0 is not expected because of the filter.
    octant = (1 if X > 0 else -1, 1 if Y > 0 else -1, 1 if Z > 0 else -1)
    
    # Calculate an angle using arctan2 for sorting (using X and Y only)
    angle = math.atan2(Y, X)
    
    candidates.append({
        'm': m,
        'n': n,
        'X': X,
        'Y': Y,
        'Z': Z,
        'octant': octant,
        'angle': angle
    })

# -----------------------------------------------------------------------------
# Group candidates by octant
# -----------------------------------------------------------------------------
octant_groups = {}
# Define all possible octants (in 3D there are 8)
all_octants = [(sx, sy, sz) for sx in [1, -1] for sy in [1, -1] for sz in [1, -1]]
for oc in all_octants:
    octant_groups[oc] = []

for cand in candidates:
    oc = cand['octant']
    octant_groups[oc].append(cand)

# -----------------------------------------------------------------------------
# Select evenly spaced candidates from each octant group
# -----------------------------------------------------------------------------
T_per_octant = T // 8
selected = {}

for oc, group in octant_groups.items():
    # Sort by the computed angle
    group.sort(key=lambda c: c['angle'])
    if len(group) < T_per_octant:
        st.warning(f"Octant {oc} has insufficient candidates. Required: {T_per_octant}, available: {len(group)}. Adjust parameters.")
        selected[oc] = group[:]  # take all we have
    else:
        # Evenly spaced selection indices
        indices = np.linspace(0, len(group) - 1, T_per_octant, dtype=int)
        sel = [group[i] for i in indices]
        selected[oc] = sel

# -----------------------------------------------------------------------------
# Enforce mirror image rule across opposite octants
# For each pair of opposite octants, if a candidate in one octant has
# an exact mirror (i.e. negated coordinates) in the opposite octant, we remove one.
# -----------------------------------------------------------------------------
final_selected = {}
# Start with a copy of the current selection per octant.
for oc, sel in selected.items():
    final_selected[oc] = sel.copy()

processed = set()
for oc in all_octants:
    mirror_oc = (-oc[0], -oc[1], -oc[2])
    # Process each mirror pair only once.
    if oc in processed or mirror_oc in processed:
        continue
    processed.add(oc)
    processed.add(mirror_oc)
    group1 = final_selected.get(oc, [])
    group2 = final_selected.get(mirror_oc, [])
    
    # Build a set of coordinates in group1 for quick lookup.
    group1_coords = {(cand['X'], cand['Y'], cand['Z']) for cand in group1}
    
    # Remove from group2 any candidate whose mirror exists in group1.
    new_group2 = []
    for cand in group2:
        mirror_candidate = (-cand['X'], -cand['Y'], -cand['Z'])
        if mirror_candidate in group1_coords:
            # Remove the candidate from the opposite octant.
            continue
        else:
            new_group2.append(cand)
    final_selected[mirror_oc] = new_group2
    # Note: If after removal an octant has fewer than T_per_octant points, you could add logic to refill it
    # from its originally ordered candidate list.

# -----------------------------------------------------------------------------
# Combine results and limit to T points (if more than T due to slight overfill)
# -----------------------------------------------------------------------------
result = []
for oc in all_octants:
    result.extend(final_selected.get(oc, []))

if len(result) < T:
    st.warning(f"Total selected points ({len(result)}) are less than T ({T}).")
elif len(result) > T:
    result = result[:T]  # trim to exactly T points

# -----------------------------------------------------------------------------
# Output final result as a DataFrame with three columns: X, Y, Z.
# -----------------------------------------------------------------------------
if result:
    df = pd.DataFrame(result)[['X', 'Y', 'Z']]
    st.write("Selected Points (X, Y, Z):")
    st.dataframe(df)
else:
    st.write("No candidate points passed the filter criteria.")
