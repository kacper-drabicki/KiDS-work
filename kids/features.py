MAGNITUDES = ['MAG_GAAP_u','MAG_GAAP_g', 'MAG_GAAP_r','MAG_GAAP_i1', 'MAG_GAAP_i2',
              'MAG_GAAP_Z', 'MAG_GAAP_Y','MAG_GAAP_J', 'MAG_GAAP_H','MAG_GAAP_Ks']

COLORS = ['u-g', 'u-r', 'u-i1', 'u-i2', 'u-Z', 'u-Y','u-J', 'u-H', 'u-Ks', 'g-r', 'g-i1', 'g-i2',
          'g-Z', 'g-Y', 'g-J', 'g-H','g-Ks', 'r-i1', 'r-i2', 'r-Z', 'r-Y', 'r-J', 'r-H', 'r-Ks', 
          'i1-i2','i1-Z', 'i1-Y', 'i1-J', 'i1-H', 'i1-Ks', 'i2-Z', 'i2-Y', 'i2-J', 'i2-H','i2-Ks', 
          'Z-Y', 'Z-J', 'Z-H', 'Z-Ks', 'Y-J', 'Y-H', 'Y-Ks', 'J-H','J-Ks', 'H-Ks']

RATIOS = ['u/g', 'u/r', 'u/i1', 'u/i2', 'u/Z', 'u/Y', 'u/J','u/H', 'u/Ks', 'g/r', 'g/i1', 'g/i2',
          'g/Z', 'g/Y', 'g/J', 'g/H', 'g/Ks', 'r/i1', 'r/i2', 'r/Z', 'r/Y', 'r/J', 'r/H', 'r/Ks',
          'i1/i2', 'i1/Z', 'i1/Y', 'i1/J', 'i1/H', 'i1/Ks', 'i2/Z', 'i2/Y', 'i2/J', 'i2/H', 'i2/Ks',
          'Z/Y', 'Z/J', 'Z/H', 'Z/Ks', 'Y/J', 'Y/H', 'Y/Ks', 'J/H', 'J/Ks', 'H/Ks']

clf_features = COLORS + RATIOS + ['CLASS_STAR']
photoz_features = MAGNITUDES + COLORS

