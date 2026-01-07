#!/usr/bin/env python3
"""
Generate Publication Figures from V9 Data
Uses existing results from CODIGO_6_V9_SENSITIVITY
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from PIL import Image
import os
import json

# Publication settings
DPI = 600

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
rcParams['font.size'] = 9
rcParams['axes.labelsize'] = 10
rcParams['axes.titlesize'] = 10
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['legend.fontsize'] = 8
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = DPI
rcParams['axes.linewidth'] = 0.8
rcParams['lines.linewidth'] = 1.2
rcParams['mathtext.fontset'] = 'stix'

OUTPUT_DIR = '/Users/yatrogenesis/Desktop/PAPER_FINAL/figures'
V9_DIR = '/Users/yatrogenesis/Desktop/CODIGO_6_V9_SENSITIVITY'


def save_all_formats(fig, basename):
    """Save in PDF, EPS, PNG, and high-quality JPG."""
    # PDF
    fig.savefig(f'{OUTPUT_DIR}/{basename}.pdf', format='pdf', dpi=DPI,
                bbox_inches='tight', facecolor='white')
    print(f"   {basename}.pdf")

    # EPS
    fig.savefig(f'{OUTPUT_DIR}/{basename}.eps', format='eps', dpi=DPI,
                bbox_inches='tight', facecolor='white')
    print(f"   {basename}.eps")

    # PNG
    png_path = f'{OUTPUT_DIR}/{basename}.png'
    fig.savefig(png_path, format='png', dpi=DPI,
                bbox_inches='tight', facecolor='white')
    print(f"   {basename}.png")

    # JPG (convert from PNG using PIL for quality control)
    jpg_path = f'{OUTPUT_DIR}/{basename}.jpg'
    img = Image.open(png_path)
    if img.mode == 'RGBA':
        # Convert RGBA to RGB with white background
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    img.save(jpg_path, 'JPEG', quality=100, subsampling=0)
    print(f"   {basename}.jpg")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("GENERATING PUBLICATION FIGURES FROM V9 DATA")
    print("=" * 60)

    # Load V9 images and convert
    print("\n1. Converting V9 figures to publication format...")

    # Read existing V9 PNG figures and convert
    v9_files = {
        'fig1_ensemble_dynamics': 'FIG1_ensemble_with_derivative.png',
        'fig2_validation_gaps': 'FIG2_gap_comparison.png',
        'fig3_mechanism': 'FIG3_individual_trials.png'
    }

    for out_name, v9_name in v9_files.items():
        v9_path = f'{V9_DIR}/{v9_name}'
        if os.path.exists(v9_path):
            print(f"\n   Processing {v9_name}...")
            img = Image.open(v9_path)

            # Save in all formats
            # PDF
            img_rgb = img.convert('RGB') if img.mode == 'RGBA' else img
            img_rgb.save(f'{OUTPUT_DIR}/{out_name}.pdf', 'PDF', resolution=DPI)
            print(f"   {out_name}.pdf")

            # PNG (high res)
            img.save(f'{OUTPUT_DIR}/{out_name}.png', 'PNG')
            print(f"   {out_name}.png")

            # JPG
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img_rgb = background
            img_rgb.save(f'{OUTPUT_DIR}/{out_name}.jpg', 'JPEG', quality=100, subsampling=0)
            print(f"   {out_name}.jpg")

            # EPS (via matplotlib)
            fig, ax = plt.subplots(figsize=(img.width/DPI, img.height/DPI), dpi=DPI)
            ax.imshow(np.array(img))
            ax.axis('off')
            fig.savefig(f'{OUTPUT_DIR}/{out_name}.eps', format='eps', dpi=DPI,
                       bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"   {out_name}.eps")

    # Generate clean versions with proper formatting
    print("\n2. Generating clean publication figures...")

    # Load V9 summary
    with open(f'{V9_DIR}/results_summary.json', 'r') as f:
        v9_summary = json.load(f)

    print(f"\n   V9 Results:")
    print(f"   - Crystal: {v9_summary['results']['n_crystal']}/30")
    print(f"   - CUSUM precursor rate: {v9_summary['results']['cusum_method']['precursor_rate']*100:.1f}%")
    print(f"   - Mean gap: {v9_summary['results']['cusum_method']['mean_gap']:.1f} steps")

    # Create summary JSON
    summary = {
        'source': 'V9 CUSUM Analysis',
        'n_trials': 30,
        'n_crystal': v9_summary['results']['n_crystal'],
        'cusum_precursor_rate': v9_summary['results']['cusum_method']['precursor_rate'],
        'cusum_mean_gap': v9_summary['results']['cusum_method']['mean_gap'],
        'cusum_std_gap': v9_summary['results']['cusum_method']['std_gap'],
        'derivative_precursor_rate': v9_summary['results']['derivative_method']['precursor_rate'],
        'figures_generated': ['fig1_ensemble_dynamics', 'fig2_validation_gaps', 'fig3_mechanism'],
        'formats': ['pdf', 'eps', 'png', 'jpg'],
        'dpi': DPI
    }

    with open(f'{OUTPUT_DIR}/figure_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_DIR}")
    print("\nFiles:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(f'{OUTPUT_DIR}/{f}')
        print(f"   {f}: {size/1024:.1f} KB")
