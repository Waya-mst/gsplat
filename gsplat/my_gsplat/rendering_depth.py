"""
render_depth.py
===============
gsplat の学習済みチェックポイント (.pt) から、
任意のカメラ位置・方向・画角を指定して深度マップをレンダリングするスクリプト。

使い方:
    python render_depth.py \
        --ckpt results/garden/ckpts/ckpt_29999.pt \
        --position 0.0 0.0 3.0 \
        --look_at  0.0 0.0 0.0 \
        --up       0.0 1.0 0.0 \
        --fov_deg  60.0 \
        --width    1280 \
        --height   720 \
        --output   depth.png
"""

import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ─────────────────────────────────────────────
# カメラ行列ユーティリティ
# ─────────────────────────────────────────────

def look_at_to_c2w(position: np.ndarray,
                   look_at: np.ndarray,
                   up: np.ndarray) -> np.ndarray:
    """
    カメラ位置・注視点・上方向ベクトルから
    camera-to-world (c2w) 行列 [4, 4] を生成する。

    gsplat は OpenCV 座標系 (Z 前方) を使用。
    """
    position = np.asarray(position, dtype=np.float64)
    look_at  = np.asarray(look_at,  dtype=np.float64)
    up       = np.asarray(up,       dtype=np.float64)

    forward = look_at - position
    forward = forward / np.linalg.norm(forward)          # +Z

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)                # +X

    up_corrected = np.cross(right, forward)              # +Y (下向き修正済み)

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = -up_corrected   # OpenCV: Y 軸は下向き
    c2w[:3, 2] = forward
    c2w[:3, 3] = position
    return c2w


def fov_to_intrinsics(fov_deg: float,
                      width: int,
                      height: int) -> np.ndarray:
    """
    水平画角 (度) から内部行列 K [3, 3] を生成する。
    fx = fy (正方ピクセル想定)。
    """
    fov_rad = math.radians(fov_deg)
    fx = width  / (2.0 * math.tan(fov_rad / 2.0))
    fy = fx                              # 正方ピクセル
    cx = width  / 2.0
    cy = height / 2.0
    K = np.array([[fx,  0, cx],
                  [ 0, fy, cy],
                  [ 0,  0,  1]], dtype=np.float64)
    return K


# ─────────────────────────────────────────────
# チェックポイント読み込み
# ─────────────────────────────────────────────

def load_splats(ckpt_path: str, device: str):
    """
    simple_trainer.py が保存した .pt チェックポイントから
    Gaussian パラメータを読み込み、活性化関数を適用して返す。

    Returns:
        means      [N, 3]  – ガウシアン中心 (world 座標)
        quats      [N, 4]  – 正規化クォータニオン (w, x, y, z)
        scales     [N, 3]  – スケール (exp 適用済み)
        opacities  [N]     – 不透明度 (sigmoid 適用済み)
        sh0        [N, 1, 3] – SH 係数 (degree-0)
        shN        [N, K, 3] – SH 係数 (degree 1+)、なければ None
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    # ckpt の構造: {"splats": {...}, "step": int, ...}
    splats = ckpt.get("splats", ckpt)   # 旧形式との互換

    means     = splats["means"].to(device)
    quats     = F.normalize(splats["quats"], dim=-1).to(device)
    scales    = torch.exp(splats["scales"]).to(device)
    opacities = torch.sigmoid(splats["opacities"]).to(device)
    sh0       = splats["sh0"].to(device)
    shN       = splats.get("shN", None)
    if shN is not None:
        shN = shN.to(device)

    print(f"[load_splats] N = {means.shape[0]:,} Gaussians loaded from '{ckpt_path}'")
    return means, quats, scales, opacities, sh0, shN


# ─────────────────────────────────────────────
# レンダリング
# ─────────────────────────────────────────────

def render_depth(
    ckpt_path : str,
    position  : list,
    look_at   : list,
    up        : list,
    fov_deg   : float,
    width     : int,
    height    : int,
    output    : str,
    near      : float = 0.01,
    far       : float = 1e10,
    mode      : str   = "ED",   # "D" or "ED"
    device    : str   = "cuda",
    colormap  : bool  = True,
) -> None:
    """
    メインレンダリング関数。
    """
    from gsplat import rasterization

    # ── 1. Gaussian 読み込み ──
    means, quats, scales, opacities, sh0, shN = load_splats(ckpt_path, device)

    # SH 係数を結合 (sh_degree=3 想定)
    if shN is not None:
        colors = torch.cat([sh0, shN], dim=1)   # [N, K, 3]
        sh_degree = int(round(math.sqrt(colors.shape[1]))) - 1
    else:
        colors = sh0                             # [N, 1, 3]
        sh_degree = 0

    # ── 2. カメラ行列構築 ──
    c2w = look_at_to_c2w(position, look_at, up)          # [4, 4]
    K   = fov_to_intrinsics(fov_deg, width, height)      # [3, 3]

    # gsplat は viewmat (w2c) を要求する
    viewmat = np.linalg.inv(c2w)                         # [4, 4]

    viewmats_t = torch.from_numpy(viewmat).float().unsqueeze(0).to(device)  # [1, 4, 4]
    Ks_t       = torch.from_numpy(K).float().unsqueeze(0).to(device)        # [1, 3, 3]

    print(f"[render] mode={mode}, fov={fov_deg}°, size={width}x{height}")
    print(f"         position={position}, look_at={look_at}")

    # ── 3. ラスタライズ ──
    with torch.no_grad():
        renders, alphas, _ = rasterization(
            means      = means,
            quats      = quats,
            scales     = scales,
            opacities  = opacities,
            colors     = colors,
            viewmats   = viewmats_t,
            Ks         = Ks_t,
            width      = width,
            height     = height,
            near_plane = near,
            far_plane  = far,
            sh_degree  = sh_degree,
            render_mode= mode,       # "D" or "ED"
        )
    # renders: [1, H, W, 1]

    depth = renders[0, ..., 0]   # [H, W]

    # ── 4. 保存 ──
    _save_depth(depth, alphas[0, ..., 0], output, colormap)


def _save_depth(depth: torch.Tensor,
                alpha: torch.Tensor,
                output: str,
                colormap: bool) -> None:
    """深度マップを PNG として保存する。"""
    depth_np = depth.cpu().float().numpy()
    alpha_np = alpha.cpu().float().numpy()

    # 有効ピクセル (alpha > 0) だけで正規化
    valid = alpha_np > 0.01
    if valid.any():
        d_min = depth_np[valid].min()
        d_max = depth_np[valid].max()
    else:
        d_min, d_max = depth_np.min(), depth_np.max()

    if d_max - d_min < 1e-6:
        d_max = d_min + 1.0

    depth_norm = (depth_np - d_min) / (d_max - d_min)   # [0, 1]
    depth_norm = np.clip(depth_norm, 0.0, 1.0)

    # 背景 (alpha≒0) を白にする
    depth_norm[~valid] = 1.0

    if colormap:
        # JET カラーマップ (matplotlib 不要の簡易実装)
        img_uint8 = _apply_jet(depth_norm)
        img = Image.fromarray(img_uint8, mode="RGB")
        out_path = output if output.lower().endswith(".png") else output + ".png"
    else:
        # グレースケール 16bit PNG
        depth_uint16 = (depth_norm * 65535).astype(np.uint16)
        img = Image.fromarray(depth_uint16, mode="I;16")
        out_path = output if output.lower().endswith(".png") else output + ".png"

    img.save(out_path)
    print(f"[save] depth map saved → {out_path}")
    print(f"       depth range (metric): {d_min:.4f} ~ {d_max:.4f}")


def _apply_jet(x: np.ndarray) -> np.ndarray:
    """
    0~1 の float 配列を JET カラーマップ RGB uint8 に変換する。
    matplotlib に依存しない簡易実装。
    """
    r = np.clip(1.5 - np.abs(x * 4.0 - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(x * 4.0 - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(x * 4.0 - 1.0), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Render a depth map from a gsplat checkpoint at an arbitrary camera pose."
    )
    p.add_argument("--ckpt",      required=True,
                   help="Path to .pt checkpoint (e.g. results/garden/ckpts/ckpt_29999.pt)")

    # カメラ姿勢
    p.add_argument("--position",  nargs=3, type=float, default=[0.0, 0.0, 3.0],
                   metavar=("X", "Y", "Z"),
                   help="Camera position in world coordinates (default: 0 0 3)")
    p.add_argument("--look_at",   nargs=3, type=float, default=[0.0, 0.0, 0.0],
                   metavar=("X", "Y", "Z"),
                   help="Point the camera looks at (default: 0 0 0)")
    p.add_argument("--up",        nargs=3, type=float, default=[0.0, 1.0, 0.0],
                   metavar=("X", "Y", "Z"),
                   help="Up vector (default: 0 1 0)")

    # 画角・解像度
    p.add_argument("--fov_deg",   type=float, default=60.0,
                   help="Horizontal field of view in degrees (default: 60)")
    p.add_argument("--width",     type=int,   default=1280,
                   help="Output image width  (default: 1280)")
    p.add_argument("--height",    type=int,   default=720,
                   help="Output image height (default: 720)")

    # その他
    p.add_argument("--output",    default="depth.png",
                   help="Output PNG path (default: depth.png)")
    p.add_argument("--mode",      choices=["D", "ED"], default="ED",
                   help="D=accumulated depth, ED=expected depth (default: ED)")
    p.add_argument("--near",      type=float, default=0.01)
    p.add_argument("--far",       type=float, default=1e10)
    p.add_argument("--no_color",  action="store_true",
                   help="Save as 16-bit grayscale instead of JET colormap")
    p.add_argument("--cpu",       action="store_true",
                   help="Force CPU (very slow, for debugging)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    render_depth(
        ckpt_path = args.ckpt,
        position  = args.position,
        look_at   = args.look_at,
        up        = args.up,
        fov_deg   = args.fov_deg,
        width     = args.width,
        height    = args.height,
        output    = args.output,
        near      = args.near,
        far       = args.far,
        mode      = args.mode,
        device    = device,
        colormap  = not args.no_color,
    )