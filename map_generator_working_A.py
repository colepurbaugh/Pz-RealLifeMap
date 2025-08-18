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

def generate_map_grid_center(lat, lon, cells_x, cells_y, road_width_scale, margin_factor, status_label):
    try:
        status_label.config(text="Downloading OSM data... Please wait.", fg="orange")
        print("Step 1: Downloading OSM data...")
        root.update()

        cell_size_m = 300
        cell_px = 300
        total_zone_w_m = cell_size_m * cells_x
        total_zone_h_m = cell_size_m * cells_y
        margin_w_m = total_zone_w_m * margin_factor
        margin_h_m = total_zone_h_m * margin_factor
        download_w_m = total_zone_w_m + 2 * margin_w_m
        download_h_m = total_zone_h_m + 2 * margin_h_m
        dist = 0.5 * math.sqrt(download_w_m ** 2 + download_h_m ** 2)

        print(f"Rendering zone: {total_zone_w_m}m x {total_zone_h_m}m")
        print(f"Margin applied: {margin_w_m}m (W), {margin_h_m}m (H)")
        print(f"Download zone: {download_w_m}m x {download_h_m}m (radius {dist:.1f}m)")

        dpi = 100
        plt.rcParams['path.simplify'] = False
        plt.rcParams['agg.path.chunksize'] = 0
        plt.rcParams['lines.antialiased'] = False

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

        print("Downloading road network...")
        G = ox.graph_from_point((lat, lon), dist=dist, network_type='all',
                                simplify=False, retain_all=True, truncate_by_edge=True)
        gdf_edges = ox.graph_to_gdfs(G, nodes=False)
        print(f"Roads downloaded: {len(gdf_edges)}")

        status_label.config(text="Downloading map features...", fg="orange")
        print("Step 2: Downloading map features...")
        root.update()

        try:
            gdf_features = ox.features_from_point((lat, lon), tags=tags, dist=dist)
            print(f"Features downloaded: {len(gdf_features)}")
        except Exception as e:
            print(f"Warning: Some features could not be downloaded: {e}")
            simple_tags = {'natural': True, 'landuse': True, 'highway': True, 'waterway': True}
            gdf_features = ox.features_from_point((lat, lon), tags=simple_tags, dist=dist)

        status_label.config(text="Projecting geometries...", fg="orange")
        print("Step 3: Projecting geometries...")
        root.update()

        gdf_edges_utm = ox.projection.project_gdf(gdf_edges)
        utm_crs = gdf_edges_utm.crs
        gdf_features_utm = gdf_features.to_crs(utm_crs)

        center_x = gdf_edges_utm.geometry.centroid.x.mean()
        center_y = gdf_edges_utm.geometry.centroid.y.mean()
        print(f"Map center: ({center_x:.0f}, {center_y:.0f})")

        total_map_px_x = cell_px * cells_x
        total_map_px_y = cell_px * cells_y
        meters_per_pixel = total_zone_w_m / total_map_px_x if total_map_px_x > 0 else 1.0

        dpi = 100
        plt.rcParams['path.simplify'] = False
        plt.rcParams['agg.path.chunksize'] = 0
        plt.rcParams['lines.antialiased'] = False
        fig, ax = plt.subplots(figsize=(total_map_px_x / dpi, total_map_px_y / dpi), dpi=dpi)
        fig.subplots_adjust(0, 0, 1, 1)

        xmin = center_x - total_zone_w_m / 2
        xmax = center_x + total_zone_w_m / 2
        ymin = center_y - total_zone_h_m / 2
        ymax = center_y + total_zone_h_m / 2

        print(f"Rendering zone: x[{xmin:.0f}, {xmax:.0f}], y[{ymin:.0f}, {ymax:.0f}]")

        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                               facecolor=PALETTE['light_grass'], edgecolor='none', zorder=0))

        # Draw features
        layer = gdf_features_utm[gdf_features_utm.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
        if not layer.empty:
            print(f"Total land objects: {len(layer)}")
            for col, func in [('natural', get_natural_color), ('landuse', get_landuse_color)]:
                if col in layer.columns:
                    colors = layer[col].apply(func)
                    mask = colors.notna()
                    if col == 'natural':
                        water_mask = layer[col].astype(str).str.lower().isin(['water', 'wetland', 'bay', 'coastline'])
                        mask = mask & ~water_mask
                    if mask.any():
                        print(f"Rendering {col}: {mask.sum()} objects")
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

        print(f"Sorting and drawing roads: {len(gdf_edges_utm)} roads")
        roads_to_draw = []
        for idx, row in gdf_edges_utm.iterrows():
            highway = row.get('highway')
            if highway:
                priority = get_road_priority(highway)
                roads_to_draw.append((priority, idx, row))
        roads_to_draw.sort(key=lambda x: x[0])

        print(f"Roads sorted by priority, drawing {len(roads_to_draw)} roads...")
        routes_drawn = 0
        for priority, idx, row in roads_to_draw:
            highway = row.get('highway')
            surface = row.get('surface')
            color = get_road_color(highway, surface)
            width_m = get_road_width_m(highway)
            lw = max((width_m / meters_per_pixel) * 0.01 * road_width_scale, 0.1)
            try:
                x, y = row.geometry.xy
                ax.plot(x, y, color=color, linewidth=lw, solid_capstyle='round', zorder=4)
                routes_drawn += 1
            except Exception as e:
                print(f"Error drawing a road: {e}")
                continue

        print(f"Roads drawn: {routes_drawn}")

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_axis_off()
        ax.set_aspect('equal')

        status_label.config(text="Saving complete map image...", fg="orange")
        print("Step 5: Saving complete map image...")
        root.update()

        complete_map_filename = "complete_map.png"
        for artist in ax.get_children():
            if hasattr(artist, 'set_antialiased'):
                artist.set_antialiased(False)

        plt.savefig(complete_map_filename, dpi=dpi, pad_inches=0, bbox_inches='tight')
        plt.close()
        print(f"Complete map saved: {complete_map_filename}")

        status_label.config(text="Processing complete vegetation map...", fg="orange")
        print("Step 5.5: Processing complete vegetation map...")
        root.update()

        img = Image.open(complete_map_filename).convert("RGB")
        img_array = np.array(img)
        veg_array = classify_vegetation_color_vectorized(img_array)
        complete_veg_filename = "complete_vegetation_map.png"
        Image.fromarray(veg_array).save(complete_veg_filename)
        print(f"Complete vegetation map saved: {complete_veg_filename}")

        status_label.config(text="Slicing maps into cells...", fg="orange")
        print("Step 6: Slicing maps into cells...")
        root.update()

        for row in range(cells_y):
            for col in range(cells_x):
                y_start, y_end = row * cell_px, (row + 1) * cell_px
                x_start, x_end = col * cell_px, (col + 1) * cell_px
                cell_img = img_array[y_start:y_end, x_start:x_end]
                Image.fromarray(cell_img).save(f"{output_dir}/{col},{row}.png")
                veg_cell_img = veg_array[y_start:y_end, x_start:x_end]
                Image.fromarray(veg_cell_img).save(f"{veg_output_dir}/{col},{row}_veg.png")

        status_label.config(text=f"{cells_x}x{cells_y} grids generated. Complete images saved.", fg="#004d00")
        print(f"Map grid generation completed: {cells_x}x{cells_y} tiles saved in '{output_dir}' and '{veg_output_dir}'.")

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

lat_entry = add_entry("Latitude (center):", 47.39000, 0, "Latitude of the map center point.")
lon_entry = add_entry("Longitude (center):", -122.45400, 1, "Longitude of the map center point.")
cells_x_entry = add_entry("Cells X (width):", 6, 2, "Number of cells horizontally.")
cells_y_entry = add_entry("Cells Y (height):", 5, 3, "Number of cells vertically.")
margin_entry = add_entry("Download margin (fraction):", 0.25, 4,
                         "Extra area to download around the map zone (fraction).")
width_entry = add_entry("Road width scale:", 100, 5,
                        "Scale factor for road widths on the generated maps.")

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
                f"This will generate {cells_x}x{cells_y} = {total_cells} cells, which exceeds the recommended {CELLS_WARNING_THRESHOLD}.\nDo you really wish to continue?"
            )
            if not proceed:
                status_label.config(text="Generation cancelled by user.", fg="red")
                return
        generate_map_grid_center(lat, lon, cells_x, cells_y, width_scale, margin, status_label)
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

tk.Button(frame, text="Generate Maps + Vegetation", command=on_generate_maps).grid(row=6, column=0, columnspan=2, pady=5)
tk.Button(frame, text="Preview", command=on_preview).grid(row=7, column=0, columnspan=2, pady=2)

status_label = tk.Label(root, text="", fg="green")
status_label.pack(pady=5)

root.mainloop()