import os
import shutil
import tkinter as tk
from tkinter.messagebox import showerror, askyesno, showwarning
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
# Allow large images generated locally without Pillow's decompression bomb limit
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import osmnx as ox
import math
import geopandas as gpd
from shapely.geometry import box as shapely_box
import webbrowser
import time
import io

# ========== PALETTES ==========
PALETTE = {
    'dark_grass': (90/255, 100/255, 35/255),
    'medium_grass': (117/255, 117/255, 47/255),
    'light_grass': (145/255, 135/255, 60/255),
    'sand': (210/255, 200/255, 160/255),
    'water': (0/255, 138/255, 255/255),
    'dark_asphalt': (100/255, 100/255, 100/255),
    'medium_asphalt': (120/255, 120/255, 120/255),
    'light_asphalt': (165/255, 160/255, 140/255),
    'gravel_dirt': (140/255, 70/255, 15/255),
    'dirt': (120/255, 70/255, 20/255),
}

PALETTE_ORIG = {
    'dark_grass': (90, 100, 35),
    'medium_grass': (117, 117, 47),
    'light_grass': (145, 135, 60),
}

VEGETATION_COLORS = {
    'dense_trees_and_dark_grass': (127, 0, 0),
    'trees_and_grass': (64, 0, 0),
    'light_long_grass': (0, 255, 0),
}

# Maximum recommended total cells before warning
CELLS_WARNING_THRESHOLD = 1600

# ========== UTILS ==========
def compute_utm_crs(lat, lon):
    """Return an EPSG code string for a suitable UTM CRS for the given lat/lon."""
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"

def clean_gdf_geometries(gdf):
    """Drop null/empty geometries. Only try to fix invalid polygons; leave lines/points unchanged."""
    if gdf is None or gdf.empty:
        return gdf
    gdf = gdf[gdf.geometry.notnull()].copy()
    def fix_geom(geom):
        if geom is None:
            return None
        geom_type = getattr(geom, 'geom_type', '')
        if geom_type in ('Polygon', 'MultiPolygon'):
            try:
                from shapely.validation import make_valid as shapely_make_valid
                return shapely_make_valid(geom)
            except Exception:
                try:
                    return geom.buffer(0)
                except Exception:
                    return None
        # Do not alter LineString/MultiLineString/Point types
        return geom
    gdf['geometry'] = gdf['geometry'].apply(fix_geom)
    gdf = gdf[gdf.geometry.notnull()]
    if hasattr(gdf.geometry, 'is_empty'):
        gdf = gdf[~gdf.geometry.is_empty]
    return gdf

def get_natural_color(natural_value):
    if natural_value is None:
        return None
    nv = str(natural_value).lower()
    if nv in ['tree', 'wood', 'shrubbery', 'tree_row', 'forest']:
        return PALETTE['dark_grass']
    elif nv in ['grassland', 'heath', 'scrub', 'meadow']:
        return PALETTE['light_grass']
    elif nv in ['fell', 'tundra']:
        return PALETTE['medium_grass']
    elif nv in ['sand', 'beach']:
        return PALETTE['sand']
    elif nv in ['water', 'wetland', 'bay', 'coastline']:
        return PALETTE['water']
    else:
        return None

def get_landuse_color(landuse_value):
    if landuse_value is None:
        return None
    lv = str(landuse_value).lower()
    if lv in ['forest', 'wood']:
        return PALETTE['dark_grass']
    elif lv in ['grass', 'meadow', 'farmland', 'recreation_ground']:
        return PALETTE['light_grass']
    return None

def get_road_color(highway, surface=None):
    if isinstance(highway, list): highway = highway[0]
    if isinstance(surface, list): surface = surface[0]
    if highway in ['motorway', 'primary', 'trunk']:
        return PALETTE['dark_asphalt']
    elif highway in ['secondary', 'tertiary', 'residential', 'service', 'unclassified']:
        return PALETTE['medium_asphalt']
    elif highway in ['path', 'track', 'bridleway', 'cycleway', 'footway']:
        if surface == 'sand':
            return PALETTE['sand']
        elif surface in ['gravel', 'dirt', 'earth']:
            return PALETTE['gravel_dirt']
        else:
            return PALETTE['dirt']
    return PALETTE['medium_asphalt']

def get_road_width_m(highway):
    if isinstance(highway, list): highway = highway[0]
    return {
        'motorway': 20,
        'primary': 15,
        'trunk': 15,
        'secondary': 10,
        'tertiary': 8,
        'residential': 7,
        'service': 7,
        'unclassified': 7,
        'path': 2,
        'track': 2,
        'bridleway': 2,
        'cycleway': 2,
        'footway': 2
    }.get(highway, 8)

def get_road_priority(highway):
    """Retourne la priorité de dessin d'une route (plus le chiffre est élevé, plus elle est dessinée tard)"""
    if isinstance(highway, list): 
        highway = highway[0]
    
    priority_order = {
        'path': 1,
        'track': 2,
        'footway': 3,
        'cycleway': 4,
        'bridleway': 5,
        'service': 6,
        'unclassified': 7,
        'residential': 8,
        'tertiary': 9,
        'secondary': 10,
        'primary': 11,
        'trunk': 12,
        'motorway': 13
    }
    
    return priority_order.get(highway, 5)  # valeur par défaut pour les types inconnus

# ========== OPTIMIZED VEGETATION MAPS ==========
def classify_vegetation_color_vectorized(img_array):
    ref_colors = np.array(list(PALETTE_ORIG.values()))
    veg_colors = np.array(list(VEGETATION_COLORS.values()))
    h, w, c = img_array.shape
    img_flat = img_array.reshape(-1, 3)
    distances = np.sqrt(np.sum((img_flat[:, np.newaxis, :] - ref_colors[np.newaxis, :, :]) ** 2, axis=2))
    closest_indices = np.argmin(distances, axis=1)
    min_distances = np.min(distances, axis=1)
    valid_mask = min_distances < 17
    result = np.zeros((h * w, 3), dtype=np.uint8)
    result[valid_mask] = veg_colors[closest_indices[valid_mask]]
    return result.reshape(h, w, 3)

def cleanup_cache():
    cache_dir = "cache"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cache directory '{cache_dir}' has been removed.")

def stitch_complete_images(output_dir, veg_output_dir, cells_x_total, cells_y_total, cell_px,
                           complete_map_filename="complete_map.png",
                           complete_veg_filename="complete_vegetation_map.png"):
    """Stitch per-tile images into complete map images."""
    try:
        width_px = cells_x_total * cell_px
        height_px = cells_y_total * cell_px
        # Background similar to light grass for safety where tiles may be missing
        bg_rgb = (145, 135, 60)
        complete_img = Image.new("RGB", (width_px, height_px), bg_rgb)
        complete_veg_img = Image.new("RGB", (width_px, height_px), (0, 0, 0))

        for row in range(cells_y_total):
            for col in range(cells_x_total):
                tile_path = os.path.join(output_dir, f"{col},{row}.png")
                veg_tile_path = os.path.join(veg_output_dir, f"{col},{row}_veg.png")
                x0 = col * cell_px
                y0 = row * cell_px
                if os.path.exists(tile_path):
                    try:
                        tile = Image.open(tile_path).convert("RGB")
                        complete_img.paste(tile, (x0, y0))
                    except Exception as e:
                        print(f"Warning: failed to paste tile {tile_path}: {e}")
                if os.path.exists(veg_tile_path):
                    try:
                        veg_tile = Image.open(veg_tile_path).convert("RGB")
                        complete_veg_img.paste(veg_tile, (x0, y0))
                    except Exception as e:
                        print(f"Warning: failed to paste veg tile {veg_tile_path}: {e}")

        complete_img.save(complete_map_filename)
        complete_veg_img.save(complete_veg_filename)
        print(f"Saved {complete_map_filename} and {complete_veg_filename}")
    except Exception as e:
        print(f"Warning: failed to stitch complete images: {e}")

def generate_map_grid_top_left(lat, lon, cells_x, cells_y, road_width_scale, margin_factor, status_label, sections=2):
    try:
        # For now, just generate bottom half (35 rows) using working logic
        actual_cells_y = 35  # Force to 35 rows
        start_row = 35  # Start from row 35 (bottom half)
        total_tiles = cells_x * actual_cells_y
        status_label.config(text="Starting generation (bottom half)...", fg="orange")
        t0 = time.perf_counter()
        print(f"params: top-left lat={lat:.5f} lon={lon:.5f} cells={cells_x}x{actual_cells_y} start_row={start_row} tiles={total_tiles} margin={margin_factor}")
        print("roads/features download starting...")
        root.update()

        cell_size_m = 300
        cell_px = 300
        total_zone_w_m = cell_size_m * cells_x
        total_zone_h_m = cell_size_m * actual_cells_y
        margin_w_m = total_zone_w_m * margin_factor
        margin_h_m = total_zone_h_m * margin_factor
        download_w_m = total_zone_w_m + 2 * margin_w_m
        download_h_m = total_zone_h_m + 2 * margin_h_m
        dist = 0.5 * math.sqrt(download_w_m ** 2 + download_h_m ** 2)

        print(f"dims: zone={int(total_zone_w_m)}x{int(total_zone_h_m)}m download={int(download_w_m)}x{int(download_h_m)}m r={int(dist)}m")

        # Prepare output dirs
        output_dir = "map_cells"
        veg_output_dir = "map_vegetation"
        for dir_name in [output_dir, veg_output_dir]:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
            os.makedirs(dir_name, exist_ok=True)

        tags = {
            'natural': True, 'landuse': True, 'leisure': True,
            'tourism': True, 'amenity': True, 'building': True,
            'waterway': True, 'coastline': True, 'water': True,
            'landcover': True, 'surface': True, 'highway': True,
            'barrier': True, 'railway': True, 'place': True
        }

        # Calculate center point from top-left for working logic
        utm_crs = compute_utm_crs(lat, lon)
        tl_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([lon], [lat]), crs="EPSG:4326").to_crs(utm_crs)
        x_tl = tl_gdf.geometry.x.iloc[0]
        y_tl = tl_gdf.geometry.y.iloc[0]
        
        # Center point for the bottom half (for OSM download)
        center_x = x_tl + total_zone_w_m / 2
        center_y = y_tl - (start_row * cell_size_m + total_zone_h_m / 2)
        
        # Convert back to lat/lon for working logic
        center_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([center_x], [center_y]), crs=utm_crs).to_crs("EPSG:4326")
        center_lat = center_gdf.geometry.y.iloc[0]
        center_lon = center_gdf.geometry.x.iloc[0]

        t = time.perf_counter()
        print("roads: start")
        G = ox.graph_from_point((center_lat, center_lon), dist=dist, network_type='all',
                                simplify=False, retain_all=True, truncate_by_edge=True)
        gdf_edges = ox.graph_to_gdfs(G, nodes=False)
        print(f"roads: count={len(gdf_edges)} t={time.perf_counter()-t:.1f}s")

        status_label.config(text="Downloading map features...", fg="orange")
        print("features: start")
        root.update()

        t = time.perf_counter()
        try:
            gdf_features = ox.features_from_point((center_lat, center_lon), tags=tags, dist=dist)
            print(f"features: count={len(gdf_features)} t={time.perf_counter()-t:.1f}s")
        except Exception as e:
            print(f"features: fallback due to error='{e}'")
            simple_tags = {'natural': True, 'landuse': True, 'highway': True, 'waterway': True}
            gdf_features = ox.features_from_point((center_lat, center_lon), tags=simple_tags, dist=dist)
            print(f"features: fallback count={len(gdf_features)} t={time.perf_counter()-t:.1f}s")

        status_label.config(text="Projecting geometries...", fg="orange")
        print("project: start")
        root.update()

        t = time.perf_counter()
        gdf_edges_utm = ox.projection.project_gdf(gdf_edges)
        utm_crs = gdf_edges_utm.crs
        gdf_features_utm = gdf_features.to_crs(utm_crs)
        print(f"project: t={time.perf_counter()-t:.1f}s")

        # Use the actual bottom-half coordinates for rendering bounds
        xmin = x_tl
        xmax = x_tl + total_zone_w_m
        ymax = y_tl - (start_row * cell_size_m)
        ymin = ymax - total_zone_h_m
        print(f"render: extent=({int(xmin)},{int(xmax)})x({int(ymin)},{int(ymax)})")

        total_map_px_x = cell_px * cells_x
        total_map_px_y = cell_px * actual_cells_y
        meters_per_pixel = total_zone_w_m / total_map_px_x if total_map_px_x > 0 else 1.0

        dpi = 100
        plt.rcParams['path.simplify'] = False
        plt.rcParams['agg.path.chunksize'] = 0
        plt.rcParams['lines.antialiased'] = False
        fig, ax = plt.subplots(figsize=(total_map_px_x / dpi, total_map_px_y / dpi), dpi=dpi)
        fig.subplots_adjust(0, 0, 1, 1)

        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                               facecolor=PALETTE['light_grass'], edgecolor='none', zorder=0))

        # Draw features
        layer = gdf_features_utm[gdf_features_utm.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
        if not layer.empty:
            for col, func in [('natural', get_natural_color), ('landuse', get_landuse_color)]:
                if col in layer.columns:
                    colors = layer[col].apply(func)
                    mask = colors.notna()
                    if col == 'natural':
                        water_mask = layer[col].astype(str).str.lower().isin(['water', 'wetland', 'bay', 'coastline'])
                        mask = mask & ~water_mask
                    if mask.any():
                        layer[mask].plot(ax=ax, color=colors[mask], linewidth=0, zorder=1)

        water_layer = gdf_features_utm[gdf_features_utm.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
        if not water_layer.empty:
            water_layer['is_water'] = False
            for col in ['natural', 'waterway', 'landuse']:
                if col in water_layer.columns:
                    water_mask = water_layer[col].astype(str).str.lower().isin(['water', 'wetland', 'bay', 'reservoir'])
                    water_layer.loc[water_mask, 'is_water'] = True
            if 'is_water' in water_layer.columns:
                water_polys = water_layer[water_layer['is_water']]
                if not water_polys.empty:
                    water_polys.plot(ax=ax, color=PALETTE['water'], linewidth=0, zorder=2)

        water_lines = gdf_features_utm[gdf_features_utm.geometry.type.isin(['LineString', 'MultiLineString'])].copy()
        if not water_lines.empty:
            water_lines['is_water_line'] = False
            for col in ['natural', 'waterway']:
                if col in water_lines.columns:
                    water_line_mask = water_lines[col].astype(str).str.lower().isin(['coastline', 'river', 'stream', 'canal', 'ditch'])
                    water_lines.loc[water_line_mask, 'is_water_line'] = True
            water_line_features = water_lines[water_lines['is_water_line']]
            if not water_line_features.empty:
                for _, row in water_line_features.iterrows():
                    waterway_type = row.get('waterway', '')
                    natural_type = row.get('natural', '')
                    if natural_type == 'coastline':
                        lw = 4
                    elif waterway_type in ['river', 'canal']:
                        lw = 3
                    else:
                        lw = 2
                    try:
                        if hasattr(row.geometry, 'xy'):
                            x, y = row.geometry.xy
                            ax.plot(x, y, color=PALETTE['water'], linewidth=lw, solid_capstyle='round', zorder=2)
                    except Exception as e:
                        print(f"Error drawing a water line: {e}")

        sand_layer = gdf_features_utm[gdf_features_utm.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
        if not sand_layer.empty:
            sand_layer['is_sand'] = False
            for col in ['natural', 'landuse']:
                if col in sand_layer.columns:
                    sand_mask = sand_layer[col].astype(str).str.lower().isin(['sand', 'beach'])
                    sand_layer.loc[sand_mask, 'is_sand'] = True
            if 'is_sand' in sand_layer.columns:
                sand_polys = sand_layer[sand_layer['is_sand']]
                if not sand_polys.empty:
                    sand_polys.plot(ax=ax, color=PALETTE['sand'], linewidth=0, zorder=3)

        t = time.perf_counter()
        roads_to_draw = []
        for idx, row in gdf_edges_utm.iterrows():
            highway = row.get('highway')
            if highway:
                priority = get_road_priority(highway)
                roads_to_draw.append((priority, idx, row))
        roads_to_draw.sort(key=lambda x: x[0])
        routes_drawn = 0
        for priority, idx, row in roads_to_draw:
            highway = row.get('highway')
            surface = row.get('surface')
            color = get_road_color(highway, surface)
            width_m = get_road_width_m(highway)
            lw = max((width_m / meters_per_pixel) * road_width_scale, 0.6)
            try:
                x, y = row.geometry.xy
                ax.plot(x, y, color=color, linewidth=lw, solid_capstyle='round', zorder=4)
                routes_drawn += 1
            except Exception as e:
                print(f"Error drawing a road: {e}")
                continue
        print(f"roads-draw: drawn={routes_drawn} t={time.perf_counter()-t:.1f}s")

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_axis_off()
        ax.set_aspect('equal')

        # Render to array; optionally also save composites
        for artist in ax.get_children():
            if hasattr(artist, 'set_antialiased'):
                artist.set_antialiased(False)
        t_render = time.perf_counter()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        _img = Image.open(buf).convert('RGB')
        img_array = np.array(_img)
        h, w = img_array.shape[:2]
        print(f"render: t={time.perf_counter()-t_render:.1f}s size={w}x{h}")

        status_label.config(text="Saving complete images...", fg="orange")
        t = time.perf_counter()
        complete_map_filename = "complete_map_1.png"
        Image.fromarray(img_array).save(complete_map_filename)
        veg_array_full = classify_vegetation_color_vectorized(img_array)
        complete_veg_filename = "complete_vegetation_map_1.png"
        Image.fromarray(veg_array_full).save(complete_veg_filename)
        print(f"stitch: done t={time.perf_counter()-t:.1f}s")

        status_label.config(text="Slicing maps into cells...", fg="orange")
        t = time.perf_counter()
        print("slice: start")
        root.update()

        veg_array = classify_vegetation_color_vectorized(img_array)
        # Slice this section into tiles
        for local_row in range(actual_cells_y):
            global_row = start_row + local_row  # Start from row 35
            y_start = local_row * cell_px
            y_end = (local_row + 1) * cell_px
            for col in range(cells_x):
                x_start = col * cell_px
                x_end = (col + 1) * cell_px
                Image.fromarray(img_array[y_start:y_end, x_start:x_end]).save(f"{output_dir}/{col},{global_row}.png")
                veg_cell_img = veg_array[y_start:y_end, x_start:x_end]
                Image.fromarray(veg_cell_img).save(f"{veg_output_dir}/{col},{global_row}_veg.png")
        print(f"slice: tiles={total_tiles} t={time.perf_counter()-t:.1f}s")

        total_time = time.perf_counter() - t0
        status_label.config(text=f"{cells_x}x{actual_cells_y} grids generated (bottom half).", fg="#004d00")
        print(f"done: tiles={total_tiles} mode=bottom-half total={total_time:.1f}s out=map_cells/,map_vegetation/")

    except Exception as e:
        showerror("Error", f"Map generation error: {e}")
        status_label.config(text="Error during map generation.", fg="red")
        print(f"ERROR during map generation: {e}")

def generate_vegetation_maps(status_label):
    try:
        status_label.config(text="Starting vegetation map generation...", fg="orange")
        print("Vegetation: start processing images.")
        input_dir = "map_cells"
        output_dir = "map_vegetation"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        filenames = [f for f in os.listdir(input_dir) if f.endswith(".png")]
        total_files = len(filenames)
        print(f"Vegetation: found {total_files} files to process.")

        for idx, filename in enumerate(filenames, 1):
            status_label.config(text=f"Processing {filename} ({idx}/{total_files})...", fg="orange")
            print(f"Vegetation: processing {filename} ({idx}/{total_files})")
            root.update()

            path = os.path.join(input_dir, filename)
            img = Image.open(path).convert("RGB")
            img_array = np.array(img)
            new_array = classify_vegetation_color_vectorized(img_array)

            base, _ = os.path.splitext(filename)
            Image.fromarray(new_array).save(os.path.join(output_dir, f"{base}_veg.png"))

        status_label.config(text=f"Vegetation maps generated in '{output_dir}'.", fg="#004d00")
        print(f"Vegetation: generation completed, saved in '{output_dir}'.")

    except Exception as e:
        showerror("Error", f"Vegetation generation error: {e}")
        status_label.config(text="Error during vegetation generation.", fg="red")
        print(f"ERROR during vegetation generation: {e}")

def show_preview(lat_n, lat_s, lon_w, lon_e, status_label):
    try:
        status_label.config(text="Generating preview...", fg="orange")
        root.update()

        cell_size_m = 300

        if not (lat_n > lat_s and lon_e > lon_w):
            raise ValueError("Lat N must be > Lat S and Lon E must be > Lon W")

        lat_center = (lat_n + lat_s) / 2.0
        lon_center = (lon_w + lon_e) / 2.0
        utm_crs = compute_utm_crs(lat_center, lon_center)

        bbox_ll = gpd.GeoDataFrame(geometry=[shapely_box(lon_w, lat_s, lon_e, lat_n)], crs="EPSG:4326")
        bbox_utm = bbox_ll.to_crs(utm_crs)
        xmin_utm, ymin_utm, xmax_utm, ymax_utm = bbox_utm.total_bounds
        width_m = xmax_utm - xmin_utm
        height_m = ymax_utm - ymin_utm

        cells_x_total = math.ceil(width_m / cell_size_m)
        cells_y_total = math.ceil(height_m / cell_size_m)
        total_cells = cells_x_total * cells_y_total
        blocks_x = math.ceil(cells_x_total / 10)
        blocks_y = math.ceil(cells_y_total / 10)
        total_blocks = blocks_x * blocks_y

        if total_cells > CELLS_WARNING_THRESHOLD:
            showwarning("Large Selection",
                        f"Selection is {cells_x_total}x{cells_y_total} = {total_cells} cells, which exceeds the recommended {CELLS_WARNING_THRESHOLD}.\nThis may be slow or unstable on some systems.")

        status_label.config(
            text=f"Preview: {cells_x_total}x{cells_y_total} cells ({total_cells} total), {total_blocks} blocks",
            fg="#004d00"
        )
        root.update()

        fig_width = 6.0
        fig_height = max(2.0, fig_width * (height_m / width_m) if width_m > 0 else 4.0)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
        fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)

        ax.add_patch(Rectangle((0, 0), width_m, height_m, facecolor=PALETTE['light_grass'], edgecolor='black', linewidth=1))

        max_grid_lines = 50
        skip_x = max(1, math.ceil(cells_x_total / max_grid_lines))
        skip_y = max(1, math.ceil(cells_y_total / max_grid_lines))

        for i in range(1, cells_x_total):
            if i % skip_x == 0:
                x = i * cell_size_m
                ax.plot([x, x], [0, height_m], color=(0.3, 0.3, 0.3), linewidth=0.4, alpha=0.4)
        for j in range(1, cells_y_total):
            if j % skip_y == 0:
                y = j * cell_size_m
                ax.plot([0, width_m], [y, y], color=(0.3, 0.3, 0.3), linewidth=0.4, alpha=0.4)

        for i in range(10, cells_x_total, 10):
            x = i * cell_size_m
            ax.plot([x, x], [0, height_m], color=(0, 0, 0), linewidth=0.8, alpha=0.8)
        for j in range(10, cells_y_total, 10):
            y = j * cell_size_m
            ax.plot([0, width_m], [y, y], color=(0, 0, 0), linewidth=0.8, alpha=0.8)

        ax.set_xlim(0, width_m)
        ax.set_ylim(0, height_m)
        ax.set_aspect('equal')
        ax.set_axis_off()

        ax.text(0.5, 1.02,
                f"Cells: {cells_x_total} x {cells_y_total}  (Total: {total_cells})\nBlocks (10x10): {blocks_x} x {blocks_y}  (Total: {total_blocks})",
                transform=ax.transAxes, ha='center', va='bottom', fontsize=9)

        try:
            plt.show(block=False)
        except TypeError:
            plt.show()

    except Exception as e:
        showerror("Error", f"Preview error: {e}")
        status_label.config(text="Error during preview.", fg="red")
        print(f"ERROR during preview: {e}")

# ========== GUI ==========
root = tk.Tk()
root.title("OSM + Vegetation Map Generator")

frame = tk.Frame(root, padx=10, pady=10)
frame.pack()

def add_entry(label_text, default_value, row, tooltip=None):
    tk.Label(frame, text=label_text).grid(row=row, column=0, sticky='e')
    entry = tk.Entry(frame, width=25)
    entry.insert(0, str(default_value))
    entry.grid(row=row, column=1)
    if tooltip:
        def on_enter(event): status_label.config(text=tooltip)
        def on_leave(event): status_label.config(text="")
        entry.bind("<Enter>", on_enter)
        entry.bind("<Leave>", on_leave)
    return entry

lat_entry = add_entry("Latitude (top boundary):", 47.515, 0, "Latitude of the top boundary of the map.")
lon_entry = add_entry("Longitude (left boundary):", -122.527, 1, "Longitude of the left boundary of the map.")
cells_x_entry = add_entry("Cells X (width):", 50, 2, "Number of cells horizontally.")
cells_y_entry = add_entry("Cells Y (height):", 70, 3, "Number of cells vertically.")
margin_entry = add_entry("Download margin (fraction):", 0.1, 4,
                         "Extra area to download around the map zone (fraction).")
width_entry = add_entry("Road width scale:", 1, 5,
                        "Scale factor for road widths on the generated maps.")
sections_entry = add_entry("Vertical Divide (sections):", 2, 6,
                        "Number of horizontal cuts (e.g., 2 = top and bottom).")

def on_generate_maps():
    try:
        lat = float(lat_entry.get())
        lon = float(lon_entry.get())
        cells_x = int(cells_x_entry.get())
        cells_y = int(cells_y_entry.get())
        margin = float(margin_entry.get())
        width_scale = float(width_entry.get())
        if cells_x < 1 or cells_y < 1:
            raise ValueError("Cells X/Y must be ≥ 1")
        if margin < 0:
            raise ValueError("Margin must be ≥ 0")
        if width_scale <= 0:
            raise ValueError("Road width scale must be > 0")
        total_cells = cells_x * cells_y
        if total_cells > CELLS_WARNING_THRESHOLD:
            proceed = askyesno(
                "Large Generation",
                f"This will generate {cells_x}x{cells_y} = {total_cells} cells, which exceeds the recommended {CELLS_WARNING_THRESHOLD}.\nDo you want to proceed?"
            )
            if not proceed:
                status_label.config(text="Generation cancelled by user.", fg="red")
                return
        sections = int(sections_entry.get()) if sections_entry.get() else 2
        if sections < 1:
            sections = 1
        generate_map_grid_top_left(lat, lon, cells_x, cells_y, width_scale, margin, status_label, sections=sections)
    except Exception as e:
        showerror("Error", f"Invalid parameter: {e}")
        status_label.config(text="Parameter error.", fg="red")

def on_generate_vegetation():
    generate_vegetation_maps(status_label)

def on_preview():
    try:
        lat = float(lat_entry.get())
        lon = float(lon_entry.get())
        cells_x = int(cells_x_entry.get())
        cells_y = int(cells_y_entry.get())
        cell_size_m = 300
        total_zone_w_m = cells_x * cell_size_m
        total_zone_h_m = cells_y * cell_size_m
        # Roughly 1e5 meters per degree lon at equator; for URL preview a coarse bbox is fine
        deg_per_m_lat = 1.0 / 111_320.0
        deg_per_m_lon = 1.0 / (111_320.0 * math.cos(math.radians(lat)) or 1e-6)
        dlat = (total_zone_h_m / 2.0) * deg_per_m_lat
        dlon = (total_zone_w_m / 2.0) * deg_per_m_lon
        lat_n = lat + dlat
        lat_s = lat - dlat
        lon_w = lon - dlon
        lon_e = lon + dlon
        url = (
            f"https://www.openstreetmap.org/export?bbox="
            f"{lon_w},{lat_s},{lon_e},{lat_n}"
        )
        webbrowser.open(url)
        status_label.config(text="Opened OSM export with current bounds.", fg="#004d00")
    except Exception as e:
        showerror("Error", f"Invalid parameter: {e}")
        status_label.config(text="Parameter error.", fg="red")

tk.Button(frame, text="Generate Maps + Vegetation", command=on_generate_maps).grid(row=8, column=0, columnspan=2, pady=5)
tk.Button(frame, text="Preview", command=on_preview).grid(row=9, column=0, columnspan=2, pady=2)

status_label = tk.Label(root, text="", fg="green")
status_label.pack(pady=5)

root.mainloop()