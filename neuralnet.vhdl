library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

-- for nearest pow of 2 (non-synth)
use IEEE.MATH_REAL.all;
use IEEE.math_real."ceil";
use IEEE.math_real."log2";

use work.nn_types.all;


entity neuralnet is
    Generic (l1i_size : integer := 8;
             l1o_size : integer := 16;
             l2i_size : integer := 16;
             l2o_size : integer := 32;
             l3i_size : integer := 32;
             l3o_size : integer := 20;
             l4i_size : integer := 20;
             l4o_size : integer := 1;
             dec_bw   : integer := 0;
             frc_bw   : integer := 15);
    Port (clk      : in STD_LOGIC;
          ena      : in STD_LOGIC;
          n_data_i : in layer_io_vector_type(0 to l1i_size - 1)(dec_bw + frc_bw + 1 - 1 downto 0);
          n_data_o : out layer_io_vector_type(0 to l4o_size - 1)(2*(dec_bw+1)+18 + frc_bw + 1 + integer(ceil(log2(real(l4i_size)))) - 1 downto 0)); -- manually computed +25
end neuralnet;

-- layer expansions: 5, 11, 18, 25

architecture Behavioral of neuralnet is
    component layer
        Generic (INPUT_SIZE     : integer;
                 OUTPUT_SIZE    : integer;
                 HAS_RELU       : std_logic;
                 DEC_BITWIDTH   : integer;
                 CARRY_DEC_BW   : integer; 
                 FRC_BITWIDTH   : integer);
        Port (clk       : in STD_LOGIC;
              ena       : in STD_LOGIC;
              done      : out STD_LOGIC;
              weights_i : in layer_weight_mtx_type(0 to INPUT_SIZE - 1, 0 to OUTPUT_SIZE - 1)(DEC_BITWIDTH + FRC_BITWIDTH - 1 + 1 downto 0); -- +1 from sign bit.
              biases_i  : in layer_bias_vector_type(0 to OUTPUT_SIZE - 1)(DEC_BITWIDTH + FRC_BITWIDTH - 1 + 1 downto 0); -- +1 from sign bit.
              data_i    : in layer_io_vector_type(0 to INPUT_SIZE - 1)(DEC_BITWIDTH + CARRY_DEC_BW + FRC_BITWIDTH + 1 - 1 downto 0); -- +1 from sign bit.
              data_o    : out layer_io_vector_type(0 to OUTPUT_SIZE - 1)(2*(DEC_BITWIDTH + 1) + CARRY_DEC_BW + FRC_BITWIDTH + 1 + integer(ceil(log2(real(INPUT_SIZE)))) - 1 downto 0)); -- last +1 from sign bit.
    end component;
    
    signal l1w : layer_weight_mtx_type(0 to l1i_size - 1, 0 to l1o_size - 1)(dec_bw + frc_bw - 1 + 1 downto 0) :=( -- +1 from sign bit.
        (x"f1f1", x"08ff", x"1f6c", x"07c1", x"c8af", x"0a9c", x"f356", x"106d", x"5428", x"2a18", x"d4be", x"2a1f", x"394e", x"d8d6", x"ea8d", x"1ff6"),
        (x"1590", x"1ada", x"2458", x"1076", x"ef67", x"dabd", x"ef7d", x"c9c1", x"1af7", x"20b6", x"26d3", x"f4c9", x"2572", x"fd66", x"241f", x"0834"),
        (x"dead", x"ecf8", x"dc20", x"259b", x"e023", x"0080", x"27f7", x"d8a4", x"f4c7", x"15d4", x"1c65", x"fd86", x"e385", x"d7b0", x"1939", x"e5e3"),
        (x"dfb4", x"f345", x"e898", x"d432", x"d0fa", x"ce67", x"2c7c", x"e1f3", x"fc89", x"1a45", x"cb34", x"088a", x"eed4", x"d696", x"d323", x"136b"),
        (x"f00d", x"d0ef", x"0cc2", x"e17c", x"e263", x"d7d8", x"fa78", x"0d07", x"06bb", x"fc83", x"0129", x"0feb", x"3178", x"f6d2", x"ee18", x"d25f"),
        (x"0f39", x"dd1c", x"2091", x"f184", x"254b", x"e104", x"ef3c", x"0a6e", x"2e10", x"c738", x"f18b", x"fefb", x"24b2", x"004a", x"fb90", x"e1ac"),
        (x"1432", x"df47", x"ea49", x"e38e", x"2fd7", x"119e", x"3bea", x"dbdd", x"e947", x"d51b", x"f1bb", x"c043", x"dbc2", x"e674", x"011d", x"e67d"),
        (x"31e9", x"e2cb", x"0c32", x"09a3", x"3775", x"0946", x"2835", x"f572", x"e41d", x"cc40", x"eb59", x"c445", x"d952", x"05ed", x"ff04", x"cef8")
    );    
    signal l1b : layer_bias_vector_type(0 to l1o_size - 1)(dec_bw + frc_bw - 1 + 1 downto 0) :=( -- +1 from sign bit.
        x"edb2", x"df83", x"fb0e", x"0c6c", x"e4fb", x"0536", x"371c", x"02cb", x"0f28", x"0c89", x"103b", x"2da9", x"23f9", x"233e", x"3187", x"ffdc"
    );
    signal l2w : layer_weight_mtx_type(0 to l2i_size - 1, 0 to l2o_size - 1)(dec_bw + frc_bw - 1 + 1 downto 0) :=( -- +1 from sign bit.
        (x"e340", x"fade", x"f9d7", x"eb23", x"1065", x"d97b", x"0145", x"e048", x"dc6c", x"f4a3", x"0582", x"fd83", x"1872", x"fb5d", x"e5f2", x"fffa", x"0c20", x"f4b7", x"1737", x"0374", x"0ebf", x"e13c", x"df66", x"f8e5", x"fe26", x"06c2", x"dd7e", x"f386", x"e2b2", x"f4fd", x"f6e3", x"ed5b"),
        (x"f0d4", x"ffbb", x"f8fd", x"073e", x"0975", x"2412", x"1fc6", x"018a", x"0e2f", x"f905", x"ee1a", x"00d7", x"263e", x"11ff", x"01ef", x"2522", x"0da2", x"f6ce", x"00f9", x"e8d0", x"15fa", x"f7cd", x"f1e7", x"fb95", x"eea2", x"2125", x"f31f", x"e810", x"2441", x"03b7", x"e031", x"03f3"),
        (x"15c9", x"edbd", x"2625", x"f350", x"06b5", x"0993", x"0b26", x"14dd", x"faa1", x"e6dc", x"10a3", x"e861", x"273f", x"1096", x"fb13", x"1df8", x"0150", x"ff57", x"d1a4", x"2594", x"09ec", x"f777", x"2485", x"11d0", x"1ce1", x"f424", x"000f", x"13d7", x"1958", x"0648", x"1c0e", x"09d7"),
        (x"ffca", x"e8a6", x"eadc", x"f2d4", x"ff22", x"07b4", x"023f", x"e780", x"e77b", x"05b9", x"10de", x"fe56", x"1d15", x"1161", x"1b37", x"13ce", x"2086", x"1a9e", x"f06f", x"1568", x"ef6e", x"142d", x"1e51", x"1180", x"1d7a", x"075b", x"13e0", x"2091", x"1240", x"ed75", x"0bde", x"01a0"),
        (x"f018", x"eb4c", x"eac2", x"e126", x"0afa", x"d179", x"e9f2", x"fb7b", x"0014", x"045e", x"f988", x"0782", x"e6d5", x"e81b", x"c7e3", x"d4e6", x"ecda", x"ff0a", x"057e", x"1676", x"ec6f", x"dda6", x"c5ea", x"dce0", x"e65c", x"e635", x"d142", x"e0c0", x"d08a", x"18ab", x"c9cc", x"eda6"),
        (x"e77a", x"057e", x"febd", x"1000", x"fecd", x"fa54", x"e1ac", x"100a", x"effd", x"f935", x"0034", x"18b6", x"f9d8", x"141c", x"f3da", x"0e5c", x"e8ec", x"e90d", x"e306", x"14b7", x"fb53", x"1bbf", x"1871", x"09aa", x"e445", x"25dc", x"22ab", x"e676", x"22b4", x"f851", x"f225", x"1e08"),
        (x"e20d", x"14c8", x"392a", x"0f2b", x"f32e", x"e9d3", x"1427", x"f16c", x"24f1", x"fec6", x"06ca", x"3406", x"3cf5", x"0db8", x"defc", x"0883", x"1627", x"fff6", x"ecfc", x"0688", x"0776", x"3900", x"32a6", x"333d", x"e701", x"07c9", x"1c8b", x"c956", x"42b2", x"079b", x"23b4", x"ed64"),
        (x"e4fd", x"1574", x"14c6", x"07e5", x"08fb", x"3522", x"ee5d", x"2a5c", x"1527", x"e5ef", x"0506", x"0490", x"1bbd", x"0ffc", x"fd0c", x"0135", x"20e6", x"0b7d", x"e6d8", x"2222", x"2400", x"1f61", x"19ff", x"061c", x"fe2e", x"ea03", x"056b", x"0eb4", x"2715", x"227f", x"0850", x"1a0e"),
        (x"f982", x"aa75", x"d3bb", x"c14c", x"2293", x"c332", x"c4f8", x"43ec", x"c313", x"15de", x"cdf9", x"d1be", x"b351", x"b0b5", x"de1e", x"afa5", x"b919", x"0fdf", x"fa82", x"daa4", x"2e9a", x"c1c0", x"e027", x"ceb1", x"f5fc", x"f5fa", x"da17", x"c3d2", x"a740", x"4ddd", x"0a1e", x"e68b"),
        (x"118c", x"1a4f", x"2b73", x"2f4d", x"f328", x"3b07", x"24ab", x"f5c3", x"09f4", x"e292", x"12be", x"07c1", x"3af6", x"feaa", x"3d5a", x"117d", x"3d3d", x"e208", x"e00d", x"2c4f", x"016b", x"1896", x"0957", x"11d2", x"e52a", x"00c5", x"2e68", x"126a", x"124b", x"14be", x"1225", x"f705"),
        (x"114c", x"1e5a", x"f950", x"ea85", x"1f61", x"0dae", x"ecfa", x"0550", x"e649", x"0781", x"0ca7", x"deab", x"115e", x"e80d", x"1595", x"e49e", x"f1ac", x"f7aa", x"f8c4", x"e870", x"0ff8", x"f6c8", x"0817", x"0edb", x"f2b3", x"0a10", x"1b19", x"0a3e", x"ebb0", x"202e", x"1382", x"10a9"),
        (x"e123", x"1673", x"1de3", x"0bd1", x"e6fb", x"1058", x"2c75", x"1b3f", x"3fb4", x"ec04", x"3854", x"0422", x"35d5", x"3d7c", x"4051", x"3bb5", x"2620", x"e1cc", x"4144", x"19c1", x"e2c8", x"2c6c", x"2ddb", x"053c", x"135e", x"166b", x"3797", x"3296", x"430e", x"1563", x"f834", x"f22d"),
        (x"13f6", x"319d", x"04ae", x"fe58", x"ff0b", x"3702", x"3db3", x"2c1e", x"28e1", x"ed3a", x"07ce", x"2776", x"02a1", x"2322", x"410e", x"3651", x"3eef", x"eff0", x"dcd9", x"f24b", x"e7a0", x"33d0", x"1022", x"0643", x"edad", x"0b3b", x"159e", x"2ed9", x"0ac9", x"fc59", x"f345", x"1a39"),
        (x"e6f5", x"0efb", x"1423", x"1a0a", x"24bd", x"eaec", x"f39b", x"11c9", x"1c97", x"0037", x"024e", x"0f42", x"0c57", x"20dd", x"f96c", x"f9dd", x"0252", x"087d", x"1a58", x"f685", x"1376", x"e9b3", x"0a47", x"f493", x"0557", x"fd95", x"ecd0", x"1b7e", x"f480", x"234a", x"ca0e", x"ea3e"),
        (x"f93c", x"e660", x"1c05", x"f811", x"113b", x"eee9", x"ecc9", x"f799", x"f714", x"e9e2", x"0f1f", x"e458", x"f7da", x"1c73", x"0eb1", x"eb8e", x"eca3", x"e682", x"efa6", x"0a21", x"f2c5", x"1abc", x"f630", x"2232", x"f6d1", x"1936", x"ff38", x"fdb1", x"13ae", x"0d38", x"ef26", x"e4d6"),
        (x"f306", x"1d07", x"217d", x"2301", x"06a4", x"1f9d", x"036d", x"e5d3", x"1588", x"01b5", x"0c5a", x"0bb5", x"224e", x"eba8", x"286a", x"237e", x"07a9", x"0909", x"07cc", x"ef44", x"1497", x"09f7", x"1a69", x"01ad", x"eeba", x"0a59", x"245d", x"ecca", x"fabb", x"e8af", x"0dec", x"f0c5")
    );    
    signal l2b : layer_bias_vector_type(0 to l2o_size - 1)(dec_bw + frc_bw - 1 + 1 downto 0) :=( -- +1 from sign bit.
        x"052c", x"22b3", x"f6b9", x"1087", x"fd70", x"fe27", x"1a95", x"ed91", x"092d", x"18a2", x"005b", x"155f", x"fb6d", x"1e66", x"e5fc", x"fcaa", x"ff55", x"077b", x"09c0", x"1d60", x"0020", x"03c6", x"1822", x"fe53", x"1a55", x"f10b", x"0194", x"019b", x"139a", x"1c1e", x"0062", x"e4f6"
    );
    signal l3w : layer_weight_mtx_type(0 to l3i_size - 1, 0 to l3o_size - 1)(dec_bw + frc_bw - 1 + 1 downto 0) :=( -- +1 from sign bit.
        (x"0b36", x"1259", x"1106", x"f378", x"f715", x"010a", x"f6ef", x"03a6", x"f71a", x"0b8f", x"fa79", x"f197", x"0b38", x"f0aa", x"fde7", x"f6b3", x"099e", x"f714", x"076c", x"0882"),
        (x"105e", x"f95e", x"1154", x"f682", x"1a32", x"0bd3", x"e87c", x"0fc5", x"f95c", x"1260", x"01ae", x"1cc9", x"0fae", x"0dd3", x"ff25", x"fb1c", x"13a9", x"fb42", x"11a5", x"eb80"),
        (x"1262", x"04cd", x"fef0", x"f962", x"073d", x"098d", x"eb21", x"1bee", x"e676", x"e642", x"0cc1", x"0328", x"2665", x"ee51", x"105b", x"0154", x"174c", x"131d", x"0ec1", x"1461"),
        (x"fc62", x"fd48", x"0309", x"e9f5", x"12e8", x"184f", x"11ff", x"12d4", x"e74c", x"0a78", x"0cd7", x"fc4a", x"0736", x"e6fc", x"e87a", x"1ae1", x"ea82", x"f190", x"1f5b", x"e5c1"),
        (x"11e4", x"f67a", x"f4ac", x"012d", x"1a23", x"fa28", x"0a85", x"ef8e", x"05a4", x"f7f8", x"0ee7", x"0e66", x"d3f1", x"0ab6", x"fe63", x"f3b2", x"e8c5", x"026c", x"162e", x"ec91"),
        (x"4022", x"2f0e", x"107d", x"0400", x"20f0", x"1be5", x"0a72", x"18b9", x"f559", x"e913", x"195a", x"2c44", x"2681", x"0055", x"092a", x"182b", x"fed8", x"106f", x"3ba4", x"0f4d"),
        (x"2724", x"3a25", x"f4a0", x"fffb", x"2d69", x"2b11", x"09f7", x"2f9f", x"ef22", x"0346", x"2103", x"323c", x"3e57", x"f825", x"f2a3", x"1b17", x"f7cc", x"1a34", x"1ed3", x"1f43"),
        (x"b1c4", x"b049", x"ee7f", x"074e", x"c746", x"b9fe", x"ead7", x"c6d4", x"f552", x"f134", x"d8ce", x"d579", x"b834", x"ef01", x"f480", x"cb03", x"1454", x"db46", x"e014", x"3da2"),
        (x"2259", x"1d8b", x"0073", x"1578", x"0c78", x"1030", x"fe12", x"1c11", x"ea8f", x"00fd", x"245a", x"faf3", x"307c", x"f00d", x"0aef", x"167e", x"fda3", x"e4a9", x"fb0b", x"1634"),
        (x"f455", x"fa8d", x"0c10", x"0604", x"ffb0", x"f91d", x"12fe", x"f03b", x"eb6a", x"effc", x"0208", x"1225", x"006c", x"084e", x"e9f5", x"04d5", x"f04b", x"0a29", x"f768", x"09f5"),
        (x"0cc9", x"02f8", x"ea20", x"f7b7", x"0702", x"1614", x"f00e", x"f047", x"09e9", x"edad", x"fa1f", x"030e", x"0f53", x"ede7", x"055c", x"01e6", x"f891", x"04a7", x"025e", x"f4e4"),
        (x"0b8c", x"1bfd", x"12e1", x"f6b0", x"fca5", x"f020", x"11db", x"1892", x"08f4", x"fc97", x"0c2f", x"fbf9", x"1b5f", x"ecbb", x"edbf", x"f876", x"f388", x"f29c", x"feb8", x"f5ce"),
        (x"fe35", x"03de", x"0699", x"f185", x"0c7c", x"14e5", x"0338", x"17b8", x"ffe6", x"f6da", x"1466", x"1618", x"20f0", x"0289", x"0a5e", x"03bb", x"0c18", x"f9b5", x"1c46", x"ec6d"),
        (x"0ee9", x"0981", x"ee8e", x"f050", x"069b", x"1c22", x"0d4a", x"f34f", x"f0a7", x"0bb0", x"f7d0", x"16a4", x"157e", x"f1e9", x"eca7", x"fd30", x"e977", x"e239", x"108e", x"f94e"),
        (x"00e5", x"03a5", x"f836", x"fc5b", x"2191", x"17da", x"f240", x"1b5d", x"030f", x"0163", x"0728", x"0059", x"31ae", x"ee46", x"0d37", x"25ee", x"fa17", x"f193", x"222b", x"268a"),
        (x"1950", x"26e2", x"121a", x"fbff", x"0978", x"0c92", x"0843", x"02cf", x"fea2", x"102d", x"19c0", x"0e2e", x"24ff", x"f872", x"ea94", x"fa45", x"ea0e", x"f04b", x"1e3a", x"0a9c"),
        (x"1081", x"1b7c", x"ed31", x"0a10", x"1bc4", x"2fd8", x"e91e", x"2f64", x"ed4a", x"0255", x"19df", x"2e37", x"28e9", x"f7a3", x"ec18", x"0c86", x"12b4", x"1f4d", x"0966", x"10f7"),
        (x"008c", x"f319", x"f822", x"f654", x"f996", x"f855", x"fcff", x"0bab", x"0337", x"f752", x"ecb0", x"0ae4", x"ed53", x"ec0d", x"0055", x"f07b", x"0b30", x"fb7e", x"ff6b", x"ef8e"),
        (x"12b2", x"53af", x"062a", x"1209", x"f13b", x"0263", x"05ee", x"7405", x"12b2", x"0e4b", x"f82e", x"18bb", x"57ec", x"0376", x"11d4", x"0248", x"f090", x"0cfe", x"07ea", x"3238"),
        (x"18ce", x"f612", x"ea4f", x"f01f", x"0eae", x"fb99", x"036e", x"ef98", x"e6aa", x"ff5e", x"0e94", x"fdc6", x"08bf", x"037e", x"ebb2", x"1758", x"086f", x"1a98", x"0b68", x"139d"),
        (x"fbd2", x"f530", x"0a1e", x"0010", x"0976", x"036e", x"0f06", x"f115", x"e5b2", x"01ed", x"fedb", x"fa22", x"eef9", x"e8d8", x"f536", x"1a66", x"0dec", x"0cbf", x"1238", x"e85a"),
        (x"2718", x"190d", x"0997", x"1457", x"0bb5", x"19c1", x"142b", x"275a", x"0148", x"f71e", x"2586", x"2951", x"35f3", x"f5c8", x"130c", x"020b", x"0689", x"f89c", x"0c27", x"f9da"),
        (x"0731", x"1bf5", x"0c6b", x"1680", x"0030", x"0adc", x"120e", x"1e7c", x"0e31", x"fe8f", x"1a39", x"1200", x"116b", x"e98b", x"fb5f", x"07a8", x"fc24", x"1009", x"0145", x"14d7"),
        (x"fc6e", x"fe74", x"edc4", x"f2e4", x"0d7b", x"0348", x"0a14", x"134c", x"ee0c", x"fba6", x"0ac5", x"1d3b", x"fbd0", x"f0c5", x"087a", x"14d3", x"056b", x"136c", x"1a7a", x"fd59"),
        (x"eda1", x"09de", x"1572", x"0407", x"fd4e", x"eac5", x"0e9f", x"0b7d", x"0395", x"fcfe", x"ff93", x"f7ed", x"e44a", x"0ad8", x"ea3b", x"0bdf", x"fc30", x"0b06", x"fb51", x"2116"),
        (x"f1de", x"1a48", x"f0d5", x"02ac", x"1505", x"0761", x"fa10", x"f377", x"0c71", x"ece0", x"ff0d", x"f7d8", x"eb40", x"f25e", x"f8e2", x"18fc", x"0705", x"deaf", x"f690", x"01e8"),
        (x"259c", x"1d37", x"12c3", x"f564", x"0eef", x"0d01", x"ea01", x"1e97", x"030b", x"fe14", x"0d20", x"0ba9", x"22a0", x"f6aa", x"f3a3", x"0d4e", x"f32a", x"15c4", x"1d3d", x"05a7"),
        (x"0858", x"0093", x"e8ef", x"f82e", x"1432", x"0f17", x"f393", x"f65d", x"0121", x"eaf8", x"109e", x"f732", x"164c", x"08c5", x"1313", x"fc74", x"0a17", x"0594", x"1b46", x"f0ab"),
        (x"0c16", x"08a2", x"e9eb", x"0620", x"12ea", x"1a66", x"fa41", x"105c", x"133d", x"f94e", x"f84a", x"129e", x"1cac", x"ff00", x"0882", x"1883", x"0e42", x"f245", x"0236", x"e9eb"),
        (x"09bf", x"fd09", x"0c47", x"f044", x"fe76", x"19fd", x"f0ec", x"f47b", x"0b88", x"0b92", x"fe54", x"1484", x"e817", x"18af", x"ee97", x"fc61", x"f176", x"0451", x"ff6b", x"0bbb"),
        (x"2c02", x"0cfc", x"0219", x"03bb", x"23f8", x"0cca", x"0b91", x"11ff", x"f296", x"f044", x"082b", x"207e", x"31db", x"f9f3", x"02f1", x"2644", x"28d2", x"341d", x"09cc", x"f322"),
        (x"1202", x"feb5", x"014e", x"09ca", x"0f67", x"fd4b", x"07f1", x"02b4", x"01f4", x"f0bf", x"0c97", x"fc77", x"0f64", x"f512", x"ffbd", x"f920", x"1639", x"ef8a", x"ebcf", x"f798")
    );    
    signal l3b : layer_bias_vector_type(0 to l3o_size - 1)(dec_bw + frc_bw - 1 + 1 downto 0) :=( -- +1 from sign bit.
        x"000b", x"f9fe", x"f7b4", x"ec93", x"15be", x"0e0a", x"f229", x"fef1", x"074e", x"f624", x"0adf", x"0b52", x"ec8c", x"09c4", x"0086", x"1919", x"f6dc", x"fee9", x"1866", x"e3e4"
    );
    -- 0 => due to super-weird VHDL rule, see https://stackoverflow.com/questions/35359413/2d-unconstrained-nx1-array/35362198#35362198
    signal l4w : layer_weight_mtx_type(0 to l4i_size - 1, 0 to l4o_size - 1)(dec_bw + frc_bw - 1 + 1 downto 0) :=( -- +1 from sign bit.
        (0 => x"0ee7"),
        (0 => x"2faa"),
        (0 => x"f45e"),
        (0 => x"1b70"),
        (0 => x"21f5"),
        (0 => x"23f3"),
        (0 => x"ffac"),
        (0 => x"2e78"),
        (0 => x"f5b2"),
        (0 => x"f77d"),
        (0 => x"1628"),
        (0 => x"0e5c"),
        (0 => x"408a"),
        (0 => x"3416"),
        (0 => x"eebb"),
        (0 => x"194b"),
        (0 => x"1e8a"),
        (0 => x"2714"),
        (0 => x"15ff"),
        (0 => x"c4f8")
    );    
    signal l4b : layer_bias_vector_type(0 to l4o_size - 1)(dec_bw + frc_bw - 1 + 1 downto 0) :=( -- +1 from sign bit.
        0 => x"ee0e"
    );
    constant l1_dec_expansion : integer := (2*(dec_bw + 1) + 0 + frc_bw + 1 + integer(ceil(log2(real(l1i_size))))) - (dec_bw + frc_bw + 1);
    constant l2_dec_expansion : integer := (2*(dec_bw + 1) + l1_dec_expansion + frc_bw + 1 + integer(ceil(log2(real(l2i_size))))) - (dec_bw + frc_bw + 1);
    constant l3_dec_expansion : integer := (2*(dec_bw + 1) + l2_dec_expansion + frc_bw + 1 + integer(ceil(log2(real(l3i_size))))) - (dec_bw + frc_bw + 1);
    signal l1out : layer_io_vector_type(0 to l1o_size - 1)((2*(dec_bw + 1) + 0 + frc_bw + 1 + integer(ceil(log2(real(l1i_size))))) - 1 downto 0);
    signal l2out : layer_io_vector_type(0 to l2o_size - 1)((2*(dec_bw + 1) + l1_dec_expansion + frc_bw + 1 + integer(ceil(log2(real(l2i_size))))) - 1 downto 0);
    signal l3out : layer_io_vector_type(0 to l3o_size - 1)((2*(dec_bw + 1) + l2_dec_expansion + frc_bw + 1 + integer(ceil(log2(real(l3i_size))))) - 1 downto 0);

    type module_state is (s_idle, s_lyr1, s_lyr2, s_lyr3, s_lyr4);
    signal ms_t0 : module_state := s_idle;
    signal ms_t1 : module_state := s_idle;
    signal ena_l1 : std_logic := '0';
    signal ena_l2 : std_logic := '0';
    signal ena_l3 : std_logic := '0';
    signal ena_l4 : std_logic := '0';
    signal ena_d    : std_logic := '0';
    signal ena_d_re : std_logic := '0';
    signal done_l1 : std_logic := '0';
    signal done_l2 : std_logic := '0';
    signal done_l3 : std_logic := '0';
    signal done_l4 : std_logic := '0';

begin
    l1_module: layer 
        generic map(
            INPUT_SIZE => l1i_size,
            OUTPUT_SIZE => l1o_size,
            HAS_RELU => '1',
            DEC_BITWIDTH => dec_bw,
            CARRY_DEC_BW => 0,
            FRC_BITWIDTH => frc_bw)
        port map(clk => clk, ena => ena_l1, done => done_l1, weights_i => l1w, biases_i => l1b, data_i => n_data_i, data_o => l1out);
    l2_module: layer 
        generic map(
            INPUT_SIZE => l2i_size,
            OUTPUT_SIZE => l2o_size,
            HAS_RELU => '1',
            DEC_BITWIDTH => dec_bw,
            CARRY_DEC_BW => l1_dec_expansion,
            FRC_BITWIDTH => frc_bw)
        port map(clk => clk, ena => ena_l2, done => done_l2, weights_i => l2w, biases_i => l2b, data_i => l1out, data_o => l2out);
    l3_module: layer 
        generic map(
            INPUT_SIZE => l3i_size,
            OUTPUT_SIZE => l3o_size,
            HAS_RELU => '1',
            DEC_BITWIDTH => dec_bw,
            CARRY_DEC_BW => l2_dec_expansion,
            FRC_BITWIDTH => frc_bw)
        port map(clk => clk, ena => ena_l3, done => done_l3, weights_i => l3w, biases_i => l3b, data_i => l2out, data_o => l3out);
    l4_module: layer 
        generic map(
            INPUT_SIZE => l4i_size,
            OUTPUT_SIZE => l4o_size,
            HAS_RELU => '0',
            DEC_BITWIDTH => dec_bw,
            CARRY_DEC_BW => l3_dec_expansion,
            FRC_BITWIDTH => frc_bw)
        port map(clk => clk, ena => ena_l4, done => done_l4, weights_i => l4w, biases_i => l4b, data_i => l3out, data_o => n_data_o);
    
    module_state_transition: process(clk)
    begin
        if rising_edge(clk) then
            ena_d <= ena;
            ms_t0 <= ms_t1;
        end if;
        ena_d_re <= not ena_d and ena;
    end process;
    
    module_state_machine: process(ms_t0, ena, done_l1, done_l2, done_l3, done_l4, clk)
    begin
        case ms_t0 is
            when s_idle  =>
                if(ena_d_re = '1') then
                    ms_t1 <= s_lyr1;
                else
                    ms_t1 <= s_idle;
                end if;
            when s_lyr1  =>
                ena_l1 <= '1';
                ena_l2 <= '0';
                ena_l3 <= '0';
                ena_l4 <= '0';
                if(done_l1 = '1') then
                    ms_t1 <= s_lyr2;
                else
                    ms_t1 <= s_lyr1;
                end if;
            when s_lyr2  =>
                ena_l1 <= '0';
                ena_l2 <= '1';
                ena_l3 <= '0';
                ena_l4 <= '0';
                if(done_l2 = '1') then
                    ms_t1 <= s_lyr3;
                else
                    ms_t1 <= s_lyr2;
                end if;
            when s_lyr3  =>
                ena_l1 <= '0';
                ena_l2 <= '0';
                ena_l3 <= '1';
                ena_l4 <= '0';
                if(done_l3 = '1') then
                    ms_t1 <= s_lyr4;
                else
                    ms_t1 <= s_lyr3;
                end if;
            when s_lyr4  =>
                ena_l1 <= '0';
                ena_l2 <= '0';
                ena_l3 <= '0';
                ena_l4 <= '1';
                if(done_l4 = '1') then
                    ms_t1 <= s_idle;
                else
                    ms_t1 <= s_lyr4;
                end if;
            when others =>  -- never hits this
                ms_t1 <= s_idle;
        end case;
    end process;

end Behavioral;
