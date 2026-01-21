"""
verify_merge_bug.py
验证 MolCrysKit 是否错误合并了空间重叠但逻辑独立的原子。
"""
import os
import numpy as np
from molcrys_kit.io.cif import scan_cif_disorder

def create_test_cif(filename):
    """创建一个包含两个位置重合但标签不同的原子的 CIF 文件"""
    content = """
data_test
_cell_length_a    10.0
_cell_length_b    10.0
_cell_length_c    10.0
_cell_angle_alpha 90.0
_cell_angle_beta  90.0
_cell_angle_gamma 90.0
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_disorder_group
N1A N 0.5 0.5 0.5 1
N1B N 0.5 0.5 0.5 2
"""
    with open(filename, 'w') as f:
        f.write(content)

def verify():
    cif_file = "debug_merge.cif"
    create_test_cif(cif_file)
    
    print(">>> 开始诊断原子合并逻辑...")
    try:
        info = scan_cif_disorder(cif_file)
        n_atoms = len(info.labels)
        
        print(f"输入: 2 个原子 (N1A, N1B)，坐标完全重合，Disorder Group 不同。")
        print(f"输出: 提取到 {n_atoms} 个原子。")
        
        if n_atoms == 1:
            print("\n[BUG CONFIRMED] 严重错误：两个不同的原子被错误合并了！")
            print("原因: 去重逻辑只检查了坐标和元素，忽略了原子标签/Disorder Group。")
            print("后果: 在无序体系中，重合的原子会被误删，导致分子'断裂'。")
        elif n_atoms == 2:
            print("\n[PASS] 逻辑正常：重合原子未被合并。")
        else:
            print(f"\n[UNKNOWN] 异常数量: {n_atoms}")
            
    finally:
        if os.path.exists(cif_file):
            os.remove(cif_file)

if __name__ == "__main__":
    verify()