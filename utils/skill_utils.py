import numpy as np

mcq_skill_id_mapping = {
 115: np.array([268.]),
 125: np.array([ 939., 2049.,  165., 2087.]),
 127: np.array([258.]),
 128: np.array([1821.,  113., 1946., 1185., 1186.,  114., 2049.,   86., 2061.]),
 129: np.array([167.]),
 130: np.array([30., 19.]),
 131: np.array([29.]),
 132: np.array([2096.,   26.]),
 135: np.array([1625., 2099., 2067.,   16.,  105., 17.0]),
 138: np.array([160.]),
 141: np.array([ 277.,  448.,  568.,   97., 1830., 1909.,  567., 2068.,  108.]),
 142: np.array([54.]),
 145: np.array([2094., 2091.]),
 146: np.array([104.]),
 152: np.array([158.]),
 154: np.array([159.]),
 155: np.array([406.]),
 156: np.array([ 358., 2087.]),
 157: np.array([ 351., 2084.]),
 158: np.array([526.]),
 159: np.array([592., 590., 589., 313., 588., 107., 387., 309.,  93., 308., 597.,
        591.]),
 160: np.array([392.]),
 161: np.array([116.]),
 163: np.array([1367., 2050.,   25.]),
 164: np.array([ 234.,  250., 1403.,  857.,  720., 1012., 1827., 1013.,  836.]),
 165: np.array([31.]),
 166: np.array([508.]),
 167: np.array([132.]),
 169: np.array([188.]),
 170: np.array([91.]),
 172: np.array([60.]),
 173: np.array([120.]),
 174: np.array([282.]),
 175: np.array([121., 660.]),
 176: np.array([2031.]),
 179: np.array([2101.]),
 180: np.array([1405.]),
 181: np.array([111.]),
 182: np.array([2102.]),
 183: np.array([580.]),
 184: np.array([359.]),
 185: np.array([87.]),
 186: np.array([55.]),
 187: np.array([61.]),
 188: np.array([519.]),
 191: np.array([334.]),
 192: np.array([2020.]),
 195: np.array([400.]),
 196: np.array([1710.]),
 197: np.array([824.]),
 198: np.array([22.]),
 199: np.array([2036., 1855., 1397.]),
 202: np.array([304.]),
 203: np.array([139.]),
 204: np.array([1955.]),
 205: np.array([312.]),
 206: np.array([1971., 2062., 1058.,  666., 1968.,  669.,  876.]),
 208: np.array([122.]),
 210: np.array([221.]),
 211: np.array([1467., 1466.,  710.,   20.]),
 212: np.array([64.]),
 213: np.array([315.]),
 214: np.array([813.,  27.]),
 215: np.array([769.]),
 216: np.array([28.]),
 217: np.array([185.]),
 224: np.array([1315.]),
 225: np.array([119.]),
 228: np.array([1013.]),
 229: np.array([598.,  93.]),
 230: np.array([309., 308., 387., 107.]),
 231: np.array([226.]),
 232: np.array([ 560., 1946.,  558.,  559.,   69.,  561., 1943.,  557.,  562.,
         563.]),
 233: np.array([223.]),
 234: np.array([2101., 2031.]),
 235: np.array([305.]),
 236: np.array([54.]),
 237: np.array([250.]),
 238: np.array([2097.]),
 239: np.array([332., 145.]),
 240: np.array([250.]),
 246: np.array([np.nan]),
 259: np.array([ 49., 433.])}

SKILL_ID_TO_MCQs = {
    16: [135],
    17: [135],
    19: [130],
    20: [211],
    22: [198],
    25: [163],
    26: [132],
    27: [214],
    28: [216],
    29: [131],
    30: [130],
    31: [165],
    54: [142],
    55: [186],
    60: [172],
    61: [187],
    64: [212],
    69: [232],
    70: [172],
    76: [207],
    86: [128],
    87: [185],
    91: [170],
    93: [229],
    97: [141],
    104: [146],
    105: [135],
    107: [230],
    108: [141],
    111: [181],
    113: [128],
    114: [128],
    116: [161],
    119: [225],
    120: [173],
    121: [175],
    122: [208],
    132: [167],
    139: [203],
    153: [222],
    158: [152],
    159: [154],
    160: [138],
    165: [125],
    167: [129],
    171: [164],
    185: [217],
    188: [169],
    193: [115],
    194: [115],
    199: [222],
    216: [157],
    221: [210],
    222: [115],
    223: [233],
    226: [231],
    234: [164],
    250: [128],
    258: [127],
    268: [115],
    277: [141],
    282: [174],
    296: [114],
    304: [202],
    305: [235],
    308: [230],
    309: [230],
    312: [205],
    313: [230],
    314: [229],
    315: [213],
    334: [191],
    351: [157],
    358: [156],
    359: [184],
    386: [222],
    387: [230],
    392: [160],
    400: [195],
    406: [155],
    411: [115, 157],
    433: [164],
    448: [141],
    475: [202],
    508: [166],
    519: [188],
    526: [158],
    541: [115],
    557: [232],
    558: [232],
    559: [232],
    562: [232],
    567: [141],
    568: [141],
    580: [183],
    588: [230],
    589: [230],
    590: [230],
    591: [230],
    592: [230],
    597: [230],
    598: [229],
    660: [175],
    666: [206],
    669: [206],
    686: [223],
    706: [222],
    710: [211],
    720: [164],
    769: [215],
    813: [214],
    824: [197],
    836: [164],
    857: [164],
    876: [206],
    939: [125],
    1012: [164],
    1013: [164, 228],
    1057: [115],
    1058: [206],
    1123: [221],
    1163: [202],
    1185: [128],
    1186: [128],
    1250: [127],
    1315: [224],
    1318: [221],
    1367: [163],
    1397: [199],
    1403: [164],
    1405: [180],
    1466: [211],
    1467: [211],
    1625: [135],
    1696: [221],
    1710: [196],
    1821: [128],
    1827: [164],
    1830: [141],
    1855: [199],
    1909: [141],
    1938: [221],
    1939: [222],
    1943: [232, 142],
    1946: [128],
    1955: [204],
    1968: [206],
    1971: [206],
    2020: [192],
    2031: [176],
    2032: [195],
    2036: [199],
    2038: [221, 222],
    2049: [128, 125],
    2050: [163],
    2061: [128],
    2062: [206],
    2067: [135],
    2068: [141],
    2084: [157],
    2087: [125, 156],
    2091: [145],
    2094: [145],
    2096: [132],
    2099: [135],
    2101: [179],
    2102: [182],
    2109: [230],
    2114: [221],
    2115: [222],
    2144: [202],
    2143: [202],
    2135: [237],
    2138: [237],
    2136: [238],
    2132: [237],
    2134: [237],
    2139: [238],
    2137: [237],
    2133: [237],
    2127: [228],
}