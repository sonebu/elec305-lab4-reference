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
        (x"f1f2", x"0900", x"1f6c", x"07c2", x"c8b0", x"0a9c", x"f357", x"106d", x"5428", x"2a18", x"d4be", x"2a20", x"394e", x"d8d7", x"ea8d", x"1ff7"),
        (x"1591", x"1adb", x"2458", x"1077", x"ef68", x"dabe", x"ef7d", x"c9c1", x"1af7", x"20b6", x"26d3", x"f4c9", x"2572", x"fd67", x"2420", x"0834"),
        (x"deae", x"ecf8", x"dc21", x"259c", x"e023", x"0080", x"27f7", x"d8a5", x"f4c7", x"15d5", x"1c65", x"fd86", x"e386", x"d7b1", x"193a", x"e5e4"),
        (x"dfb4", x"f346", x"e899", x"d432", x"d0fa", x"ce68", x"2c7d", x"e1f4", x"fc8a", x"1a46", x"cb35", x"088b", x"eed5", x"d697", x"d324", x"136c"),
        (x"f00d", x"d0f0", x"0cc3", x"e17d", x"e264", x"d7d9", x"fa78", x"0d07", x"06bc", x"fc83", x"012a", x"0fec", x"3178", x"f6d2", x"ee18", x"d25f"),
        (x"0f39", x"dd1d", x"2092", x"f185", x"254c", x"e104", x"ef3d", x"0a6f", x"2e11", x"c738", x"f18b", x"fefc", x"24b3", x"004b", x"fb90", x"e1ad"),
        (x"1433", x"df48", x"ea49", x"e38e", x"2fd7", x"119e", x"3bea", x"dbde", x"e948", x"d51c", x"f1bc", x"c043", x"dbc2", x"e675", x"011d", x"e67d"),
        (x"31ea", x"e2cb", x"0c33", x"09a4", x"3775", x"0946", x"2836", x"f573", x"e41e", x"cc40", x"eb59", x"c446", x"d952", x"05ee", x"ff04", x"cef9")
    );    
    signal l1b : layer_bias_vector_type(0 to l1o_size - 1)(dec_bw + frc_bw - 1 + 1 downto 0) :=( -- +1 from sign bit.
        x"edb3", x"df84", x"fb0f", x"0c6d", x"e4fc", x"0537", x"371c", x"02cc", x"0f28", x"0c8a", x"103c", x"2da9", x"23fa", x"233f", x"3188", x"ffdd"
    );
    signal l2w : layer_weight_mtx_type(0 to l2i_size - 1, 0 to l2o_size - 1)(dec_bw + frc_bw - 1 + 1 downto 0) :=( -- +1 from sign bit.
        (x"e341", x"fade", x"f9d8", x"eb23", x"1066", x"d97c", x"0145", x"e048", x"dc6c", x"f4a3", x"0583", x"fd83", x"1872", x"fb5d", x"e5f2", x"fffb", x"0c20", x"f4b7", x"1737", x"0374", x"0ebf", x"e13c", x"df66", x"f8e6", x"fe27", x"06c3", x"dd7e", x"f386", x"e2b3", x"f4fe", x"f6e4", x"ed5b"),
        (x"f0d4", x"ffbb", x"f8fe", x"073e", x"0975", x"2412", x"1fc6", x"018a", x"0e2f", x"f906", x"ee1a", x"00d7", x"263f", x"1200", x"01f0", x"2523", x"0da3", x"f6ce", x"00fa", x"e8d1", x"15fa", x"f7cd", x"f1e7", x"fb95", x"eea2", x"2125", x"f31f", x"e810", x"2441", x"03b8", x"e031", x"03f3"),
        (x"15ca", x"edbd", x"2626", x"f350", x"06b6", x"0994", x"0b26", x"14de", x"faa2", x"e6dc", x"10a4", x"e861", x"273f", x"1097", x"fb14", x"1df8", x"0150", x"ff58", x"d1a5", x"2595", x"09ec", x"f777", x"2485", x"11d1", x"1ce1", x"f425", x"0010", x"13d8", x"1959", x"0648", x"1c0e", x"09d8"),
        (x"ffcb", x"e8a6", x"eadd", x"f2d4", x"ff23", x"07b5", x"0240", x"e780", x"e77c", x"05b9", x"10df", x"fe57", x"1d15", x"1161", x"1b38", x"13cf", x"2087", x"1a9e", x"f06f", x"1568", x"ef6e", x"142e", x"1e52", x"1181", x"1d7b", x"075b", x"13e1", x"2092", x"1241", x"ed75", x"0bdf", x"01a1"),
        (x"f018", x"eb4d", x"eac2", x"e126", x"0afb", x"d179", x"e9f2", x"fb7c", x"0014", x"045e", x"f989", x"0782", x"e6d6", x"e81c", x"c7e3", x"d4e7", x"ecda", x"ff0b", x"057f", x"1677", x"ec6f", x"dda6", x"c5ea", x"dce1", x"e65d", x"e635", x"d143", x"e0c0", x"d08a", x"18ac", x"c9cd", x"eda7"),
        (x"e77a", x"057f", x"febd", x"1000", x"fecd", x"fa55", x"e1ad", x"100a", x"effd", x"f936", x"0035", x"18b7", x"f9d9", x"141c", x"f3da", x"0e5d", x"e8ed", x"e90e", x"e307", x"14b8", x"fb53", x"1bbf", x"1871", x"09aa", x"e446", x"25dd", x"22ac", x"e677", x"22b5", x"f852", x"f225", x"1e09"),
        (x"e20e", x"14c9", x"392b", x"0f2c", x"f32f", x"e9d3", x"1427", x"f16d", x"24f1", x"fec6", x"06cb", x"3406", x"3cf6", x"0db9", x"defd", x"0884", x"1627", x"fff7", x"ecfd", x"0688", x"0777", x"3901", x"32a7", x"333e", x"e701", x"07c9", x"1c8b", x"c956", x"42b2", x"079b", x"23b4", x"ed64"),
        (x"e4fe", x"1575", x"14c7", x"07e5", x"08fb", x"3523", x"ee5e", x"2a5c", x"1527", x"e5ef", x"0506", x"0490", x"1bbd", x"0ffc", x"fd0c", x"0135", x"20e7", x"0b7d", x"e6d9", x"2222", x"2400", x"1f62", x"19ff", x"061c", x"fe2e", x"ea04", x"056c", x"0eb4", x"2715", x"2280", x"0850", x"1a0f"),
        (x"f982", x"aa76", x"d3bb", x"c14c", x"2293", x"c332", x"c4f9", x"43ed", x"c313", x"15de", x"cdfa", x"d1bf", x"b351", x"b0b6", x"de1f", x"afa6", x"b91a", x"0fe0", x"fa82", x"daa5", x"2e9b", x"c1c1", x"e028", x"ceb2", x"f5fd", x"f5fa", x"da17", x"c3d3", x"a741", x"4dde", x"0a1e", x"e68b"),
        (x"118d", x"1a4f", x"2b73", x"2f4d", x"f328", x"3b07", x"24ab", x"f5c3", x"09f5", x"e293", x"12bf", x"07c2", x"3af7", x"feaa", x"3d5a", x"117d", x"3d3d", x"e208", x"e00d", x"2c4f", x"016c", x"1897", x"0958", x"11d3", x"e52b", x"00c6", x"2e69", x"126a", x"124c", x"14be", x"1226", x"f705"),
        (x"114d", x"1e5b", x"f950", x"ea86", x"1f61", x"0daf", x"ecfb", x"0550", x"e649", x"0781", x"0ca7", x"deac", x"115e", x"e80e", x"1595", x"e49f", x"f1ac", x"f7aa", x"f8c5", x"e871", x"0ff9", x"f6c8", x"0818", x"0edc", x"f2b3", x"0a10", x"1b19", x"0a3f", x"ebb0", x"202e", x"1382", x"10a9"),
        (x"e123", x"1674", x"1de3", x"0bd2", x"e6fc", x"1059", x"2c76", x"1b40", x"3fb4", x"ec04", x"3855", x"0422", x"35d5", x"3d7c", x"4052", x"3bb6", x"2620", x"e1cc", x"4144", x"19c1", x"e2c8", x"2c6c", x"2ddc", x"053d", x"135e", x"166b", x"3797", x"3297", x"430f", x"1563", x"f834", x"f22e"),
        (x"13f6", x"319d", x"04af", x"fe58", x"ff0c", x"3703", x"3db4", x"2c1e", x"28e1", x"ed3a", x"07cf", x"2777", x"02a1", x"2322", x"410f", x"3652", x"3eef", x"eff0", x"dcd9", x"f24b", x"e7a1", x"33d0", x"1023", x"0643", x"edad", x"0b3c", x"159e", x"2eda", x"0ac9", x"fc59", x"f345", x"1a39"),
        (x"e6f6", x"0efb", x"1423", x"1a0a", x"24bd", x"eaec", x"f39c", x"11c9", x"1c98", x"0038", x"024e", x"0f42", x"0c57", x"20de", x"f96c", x"f9dd", x"0252", x"087d", x"1a58", x"f686", x"1377", x"e9b4", x"0a47", x"f494", x"0557", x"fd96", x"ecd0", x"1b7f", x"f481", x"234a", x"ca0f", x"ea3e"),
        (x"f93c", x"e660", x"1c05", x"f812", x"113c", x"eeea", x"ecca", x"f79a", x"f715", x"e9e2", x"0f20", x"e458", x"f7da", x"1c73", x"0eb2", x"eb8f", x"eca4", x"e683", x"efa6", x"0a21", x"f2c5", x"1abc", x"f631", x"2233", x"f6d2", x"1937", x"ff38", x"fdb2", x"13af", x"0d38", x"ef27", x"e4d6"),
        (x"f306", x"1d08", x"217d", x"2301", x"06a4", x"1f9e", x"036e", x"e5d4", x"1588", x"01b5", x"0c5b", x"0bb6", x"224e", x"eba8", x"286b", x"237e", x"07aa", x"090a", x"07cc", x"ef45", x"1497", x"09f7", x"1a69", x"01ad", x"eeba", x"0a59", x"245d", x"ecca", x"fabc", x"e8b0", x"0dec", x"f0c5")
    );    
    signal l2b : layer_bias_vector_type(0 to l2o_size - 1)(dec_bw + frc_bw - 1 + 1 downto 0) :=( -- +1 from sign bit.
        x"052c", x"22b4", x"f6b9", x"1087", x"fd70", x"fe27", x"1a95", x"ed92", x"092e", x"18a2", x"005b", x"1560", x"fb6d", x"1e67", x"e5fd", x"fcaa", x"ff56", x"077c", x"09c1", x"1d60", x"0020", x"03c6", x"1822", x"fe54", x"1a55", x"f10b", x"0194", x"019c", x"139b", x"1c1f", x"0063", x"e4f7"
    );
    signal l3w : layer_weight_mtx_type(0 to l3i_size - 1, 0 to l3o_size - 1)(dec_bw + frc_bw - 1 + 1 downto 0) :=( -- +1 from sign bit.
        (x"0b36", x"125a", x"1107", x"f378", x"f715", x"010b", x"f6ef", x"03a7", x"f71a", x"0b8f", x"fa7a", x"f197", x"0b38", x"f0aa", x"fde8", x"f6b4", x"099e", x"f714", x"076d", x"0882"),
        (x"105f", x"f95e", x"1154", x"f682", x"1a32", x"0bd4", x"e87c", x"0fc5", x"f95c", x"1260", x"01ae", x"1cca", x"0fae", x"0dd4", x"ff26", x"fb1c", x"13a9", x"fb42", x"11a6", x"eb81"),
        (x"1262", x"04ce", x"fef1", x"f962", x"073e", x"098e", x"eb21", x"1bef", x"e676", x"e643", x"0cc2", x"0328", x"2665", x"ee51", x"105c", x"0155", x"174d", x"131d", x"0ec2", x"1461"),
        (x"fc63", x"fd49", x"0309", x"e9f5", x"12e9", x"1850", x"11ff", x"12d4", x"e74c", x"0a78", x"0cd8", x"fc4a", x"0736", x"e6fc", x"e87a", x"1ae2", x"ea82", x"f191", x"1f5c", x"e5c2"),
        (x"11e4", x"f67b", x"f4ac", x"012d", x"1a24", x"fa29", x"0a85", x"ef8e", x"05a5", x"f7f8", x"0ee8", x"0e66", x"d3f2", x"0ab7", x"fe64", x"f3b2", x"e8c6", x"026d", x"162e", x"ec92"),
        (x"4022", x"2f0f", x"107d", x"0400", x"20f0", x"1be5", x"0a72", x"18b9", x"f559", x"e914", x"195b", x"2c45", x"2682", x"0055", x"092a", x"182b", x"fed8", x"106f", x"3ba4", x"0f4d"),
        (x"2725", x"3a26", x"f4a0", x"fffb", x"2d69", x"2b11", x"09f8", x"2f9f", x"ef22", x"0347", x"2104", x"323c", x"3e58", x"f825", x"f2a3", x"1b18", x"f7cd", x"1a34", x"1ed4", x"1f43"),
        (x"b1c5", x"b04a", x"ee80", x"074f", x"c747", x"b9fe", x"ead8", x"c6d4", x"f552", x"f135", x"d8cf", x"d57a", x"b835", x"ef02", x"f481", x"cb04", x"1454", x"db47", x"e015", x"3da2"),
        (x"225a", x"1d8c", x"0073", x"1578", x"0c79", x"1030", x"fe13", x"1c12", x"ea8f", x"00fe", x"245a", x"faf4", x"307c", x"f00e", x"0aef", x"167e", x"fda3", x"e4a9", x"fb0b", x"1634"),
        (x"f455", x"fa8d", x"0c10", x"0604", x"ffb0", x"f91d", x"12fe", x"f03c", x"eb6a", x"effd", x"0209", x"1225", x"006d", x"084e", x"e9f5", x"04d5", x"f04c", x"0a2a", x"f768", x"09f6"),
        (x"0cc9", x"02f9", x"ea21", x"f7b7", x"0703", x"1614", x"f00f", x"f048", x"09ea", x"edae", x"fa1f", x"030f", x"0f54", x"ede8", x"055d", x"01e7", x"f891", x"04a7", x"025f", x"f4e4"),
        (x"0b8d", x"1bfd", x"12e1", x"f6b0", x"fca5", x"f020", x"11db", x"1893", x"08f5", x"fc97", x"0c30", x"fbfa", x"1b5f", x"ecbc", x"edbf", x"f877", x"f388", x"f29c", x"feb9", x"f5ce"),
        (x"fe36", x"03de", x"0699", x"f186", x"0c7d", x"14e6", x"0338", x"17b9", x"ffe7", x"f6db", x"1467", x"1618", x"20f1", x"028a", x"0a5f", x"03bb", x"0c19", x"f9b6", x"1c46", x"ec6e"),
        (x"0ee9", x"0981", x"ee8e", x"f050", x"069c", x"1c23", x"0d4b", x"f34f", x"f0a8", x"0bb0", x"f7d1", x"16a5", x"157e", x"f1ea", x"eca7", x"fd30", x"e977", x"e23a", x"108f", x"f94e"),
        (x"00e6", x"03a5", x"f837", x"fc5b", x"2191", x"17db", x"f240", x"1b5d", x"0310", x"0163", x"0728", x"0059", x"31af", x"ee46", x"0d37", x"25ef", x"fa18", x"f194", x"222b", x"268a"),
        (x"1951", x"26e2", x"121a", x"fbff", x"0979", x"0c92", x"0843", x"02d0", x"fea3", x"102d", x"19c0", x"0e2e", x"2500", x"f873", x"ea95", x"fa45", x"ea0f", x"f04c", x"1e3a", x"0a9c"),
        (x"1081", x"1b7c", x"ed31", x"0a11", x"1bc5", x"2fd9", x"e91e", x"2f65", x"ed4a", x"0255", x"19e0", x"2e38", x"28e9", x"f7a3", x"ec18", x"0c87", x"12b4", x"1f4d", x"0966", x"10f8"),
        (x"008d", x"f319", x"f822", x"f654", x"f996", x"f855", x"fd00", x"0bab", x"0338", x"f753", x"ecb1", x"0ae5", x"ed54", x"ec0e", x"0055", x"f07c", x"0b30", x"fb7f", x"ff6b", x"ef8f"),
        (x"12b3", x"53af", x"062a", x"120a", x"f13b", x"0263", x"05ee", x"7406", x"12b3", x"0e4c", x"f82f", x"18bc", x"57ed", x"0376", x"11d5", x"0249", x"f090", x"0cff", x"07eb", x"3239"),
        (x"18cf", x"f613", x"ea4f", x"f01f", x"0eae", x"fb9a", x"036f", x"ef99", x"e6ab", x"ff5e", x"0e95", x"fdc6", x"08c0", x"037e", x"ebb3", x"1759", x"086f", x"1a99", x"0b69", x"139d"),
        (x"fbd3", x"f531", x"0a1f", x"0010", x"0976", x"036f", x"0f07", x"f116", x"e5b2", x"01ed", x"fedb", x"fa23", x"eef9", x"e8d9", x"f537", x"1a67", x"0dec", x"0cbf", x"1239", x"e85b"),
        (x"2718", x"190d", x"0997", x"1457", x"0bb5", x"19c2", x"142b", x"275b", x"0149", x"f71e", x"2586", x"2952", x"35f4", x"f5c8", x"130c", x"020b", x"068a", x"f89c", x"0c28", x"f9da"),
        (x"0731", x"1bf5", x"0c6c", x"1680", x"0030", x"0add", x"120f", x"1e7c", x"0e31", x"fe8f", x"1a3a", x"1201", x"116c", x"e98b", x"fb5f", x"07a9", x"fc25", x"1009", x"0146", x"14d7"),
        (x"fc6e", x"fe74", x"edc4", x"f2e5", x"0d7c", x"0349", x"0a15", x"134c", x"ee0c", x"fba6", x"0ac6", x"1d3b", x"fbd0", x"f0c6", x"087a", x"14d3", x"056c", x"136d", x"1a7b", x"fd5a"),
        (x"eda2", x"09de", x"1572", x"0408", x"fd4f", x"eac6", x"0ea0", x"0b7d", x"0396", x"fcff", x"ff94", x"f7ed", x"e44a", x"0ad8", x"ea3c", x"0bdf", x"fc31", x"0b06", x"fb52", x"2117"),
        (x"f1df", x"1a48", x"f0d5", x"02ac", x"1505", x"0762", x"fa11", x"f378", x"0c71", x"ece0", x"ff0e", x"f7d8", x"eb40", x"f25e", x"f8e2", x"18fc", x"0706", x"deb0", x"f691", x"01e8"),
        (x"259d", x"1d38", x"12c4", x"f564", x"0ef0", x"0d01", x"ea02", x"1e97", x"030c", x"fe14", x"0d20", x"0ba9", x"22a0", x"f6aa", x"f3a4", x"0d4f", x"f32b", x"15c5", x"1d3d", x"05a8"),
        (x"0858", x"0093", x"e8f0", x"f82f", x"1432", x"0f18", x"f393", x"f65d", x"0122", x"eaf8", x"109f", x"f732", x"164c", x"08c6", x"1314", x"fc75", x"0a17", x"0595", x"1b47", x"f0ab"),
        (x"0c17", x"08a3", x"e9ec", x"0621", x"12eb", x"1a67", x"fa41", x"105c", x"133e", x"f94e", x"f84a", x"129f", x"1cac", x"ff00", x"0882", x"1883", x"0e43", x"f246", x"0236", x"e9ec"),
        (x"09bf", x"fd09", x"0c47", x"f044", x"fe76", x"19fe", x"f0ed", x"f47b", x"0b89", x"0b92", x"fe54", x"1485", x"e818", x"18af", x"ee98", x"fc61", x"f176", x"0451", x"ff6b", x"0bbb"),
        (x"2c03", x"0cfd", x"021a", x"03bc", x"23f8", x"0ccb", x"0b91", x"1200", x"f296", x"f044", x"082b", x"207e", x"31db", x"f9f4", x"02f2", x"2644", x"28d3", x"341d", x"09cd", x"f322"),
        (x"1203", x"feb6", x"014e", x"09cb", x"0f67", x"fd4b", x"07f1", x"02b5", x"01f5", x"f0bf", x"0c97", x"fc78", x"0f64", x"f512", x"ffbd", x"f921", x"1639", x"ef8b", x"ebcf", x"f799")
    );    
    signal l3b : layer_bias_vector_type(0 to l3o_size - 1)(dec_bw + frc_bw - 1 + 1 downto 0) :=( -- +1 from sign bit.
        x"000c", x"f9fe", x"f7b5", x"ec94", x"15be", x"0e0a", x"f22a", x"fef1", x"074f", x"f625", x"0ae0", x"0b52", x"ec8d", x"09c4", x"0087", x"1919", x"f6dc", x"feea", x"1866", x"e3e5"
    );
    -- 0 => due to super-weird VHDL rule, see https://stackoverflow.com/questions/35359413/2d-unconstrained-nx1-array/35362198#35362198
    signal l4w : layer_weight_mtx_type(0 to l4i_size - 1, 0 to l4o_size - 1)(dec_bw + frc_bw - 1 + 1 downto 0) :=( -- +1 from sign bit.
        (0 => x"0ee7"),
        (0 => x"2fab"),
        (0 => x"f45e"),
        (0 => x"1b70"),
        (0 => x"21f5"),
        (0 => x"23f4"),
        (0 => x"ffac"),
        (0 => x"2e79"),
        (0 => x"f5b2"),
        (0 => x"f77d"),
        (0 => x"1628"),
        (0 => x"0e5d"),
        (0 => x"408b"),
        (0 => x"3417"),
        (0 => x"eebc"),
        (0 => x"194c"),
        (0 => x"1e8a"),
        (0 => x"2714"),
        (0 => x"15ff"),
        (0 => x"c4f9")
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
