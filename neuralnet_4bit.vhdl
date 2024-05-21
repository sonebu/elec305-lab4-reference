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
             frc_bw   : integer := 4);
    Port (clk      : in STD_LOGIC;
          ena      : in STD_LOGIC;
          n_data_i : in layer_io_vector_type(0 to l1i_size - 1)(dec_bw + frc_bw + 1 - 1 downto 0);
          n_data_o : out layer_io_vector_type(0 to l4o_size - 1)(2*(dec_bw+1)+18 + frc_bw + 1 + integer(ceil(log2(real(l4i_size)))) - 1 downto 0)); -- manually computed +18
end neuralnet;

-- layer expansions: 5, 11, 18, 25 (the equation in n_data_o's size becomes equal to this)

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
        (x"f", x"0", x"1", x"0", x"c", x"0", x"f", x"1", x"5", x"2", x"d", x"2", x"3", x"d", x"e", x"1"),
        (x"1", x"1", x"2", x"1", x"e", x"d", x"e", x"c", x"1", x"2", x"2", x"f", x"2", x"f", x"2", x"0"),
        (x"d", x"e", x"d", x"2", x"e", x"0", x"2", x"d", x"f", x"1", x"1", x"f", x"e", x"d", x"1", x"e"),
        (x"d", x"f", x"e", x"d", x"d", x"c", x"2", x"e", x"f", x"1", x"c", x"0", x"e", x"d", x"d", x"1"),
        (x"f", x"d", x"0", x"e", x"e", x"d", x"f", x"0", x"0", x"f", x"0", x"0", x"3", x"f", x"e", x"d"),
        (x"0", x"d", x"2", x"f", x"2", x"e", x"e", x"0", x"2", x"c", x"f", x"f", x"2", x"0", x"f", x"e"),
        (x"1", x"d", x"e", x"e", x"2", x"1", x"3", x"d", x"e", x"d", x"f", x"c", x"d", x"e", x"0", x"e"),
        (x"3", x"e", x"0", x"0", x"3", x"0", x"2", x"f", x"e", x"c", x"e", x"c", x"d", x"0", x"f", x"c")
    );    
    signal l1b : layer_bias_vector_type(0 to l1o_size - 1)(dec_bw + frc_bw - 1 + 1 downto 0) :=( -- +1 from sign bit.
        x"e", x"d", x"f", x"0", x"e", x"0", x"3", x"0", x"0", x"0", x"1", x"2", x"2", x"2", x"3", x"f"
    );
    signal l2w : layer_weight_mtx_type(0 to l2i_size - 1, 0 to l2o_size - 1)(dec_bw + frc_bw - 1 + 1 downto 0) :=( -- +1 from sign bit.
        (x"e", x"f", x"f", x"e", x"1", x"d", x"0", x"e", x"d", x"f", x"0", x"f", x"1", x"f", x"e", x"f", x"0", x"f", x"1", x"0", x"0", x"e", x"d", x"f", x"f", x"0", x"d", x"f", x"e", x"f", x"f", x"e"),
        (x"f", x"f", x"f", x"0", x"0", x"2", x"1", x"0", x"0", x"f", x"e", x"0", x"2", x"1", x"0", x"2", x"0", x"f", x"0", x"e", x"1", x"f", x"f", x"f", x"e", x"2", x"f", x"e", x"2", x"0", x"e", x"0"),
        (x"1", x"e", x"2", x"f", x"0", x"0", x"0", x"1", x"f", x"e", x"1", x"e", x"2", x"1", x"f", x"1", x"0", x"f", x"d", x"2", x"0", x"f", x"2", x"1", x"1", x"f", x"0", x"1", x"1", x"0", x"1", x"0"),
        (x"f", x"e", x"e", x"f", x"f", x"0", x"0", x"e", x"e", x"0", x"1", x"f", x"1", x"1", x"1", x"1", x"2", x"1", x"f", x"1", x"e", x"1", x"1", x"1", x"1", x"0", x"1", x"2", x"1", x"e", x"0", x"0"),
        (x"f", x"e", x"e", x"e", x"0", x"d", x"e", x"f", x"0", x"0", x"f", x"0", x"e", x"e", x"c", x"d", x"e", x"f", x"0", x"1", x"e", x"d", x"c", x"d", x"e", x"e", x"d", x"e", x"d", x"1", x"c", x"e"),
        (x"e", x"0", x"f", x"1", x"f", x"f", x"e", x"1", x"e", x"f", x"0", x"1", x"f", x"1", x"f", x"0", x"e", x"e", x"e", x"1", x"f", x"1", x"1", x"0", x"e", x"2", x"2", x"e", x"2", x"f", x"f", x"1"),
        (x"e", x"1", x"3", x"0", x"f", x"e", x"1", x"f", x"2", x"f", x"0", x"3", x"3", x"0", x"d", x"0", x"1", x"f", x"e", x"0", x"0", x"3", x"3", x"3", x"e", x"0", x"1", x"c", x"4", x"0", x"2", x"e"),
        (x"e", x"1", x"1", x"0", x"0", x"3", x"e", x"2", x"1", x"e", x"0", x"0", x"1", x"0", x"f", x"0", x"2", x"0", x"e", x"2", x"2", x"1", x"1", x"0", x"f", x"e", x"0", x"0", x"2", x"2", x"0", x"1"),
        (x"f", x"a", x"d", x"c", x"2", x"c", x"c", x"4", x"c", x"1", x"c", x"d", x"b", x"b", x"d", x"a", x"b", x"0", x"f", x"d", x"2", x"c", x"e", x"c", x"f", x"f", x"d", x"c", x"a", x"4", x"0", x"e"),
        (x"1", x"1", x"2", x"2", x"f", x"3", x"2", x"f", x"0", x"e", x"1", x"0", x"3", x"f", x"3", x"1", x"3", x"e", x"e", x"2", x"0", x"1", x"0", x"1", x"e", x"0", x"2", x"1", x"1", x"1", x"1", x"f"),
        (x"1", x"1", x"f", x"e", x"1", x"0", x"e", x"0", x"e", x"0", x"0", x"d", x"1", x"e", x"1", x"e", x"f", x"f", x"f", x"e", x"0", x"f", x"0", x"0", x"f", x"0", x"1", x"0", x"e", x"2", x"1", x"1"),
        (x"e", x"1", x"1", x"0", x"e", x"1", x"2", x"1", x"3", x"e", x"3", x"0", x"3", x"3", x"4", x"3", x"2", x"e", x"4", x"1", x"e", x"2", x"2", x"0", x"1", x"1", x"3", x"3", x"4", x"1", x"f", x"f"),
        (x"1", x"3", x"0", x"f", x"f", x"3", x"3", x"2", x"2", x"e", x"0", x"2", x"0", x"2", x"4", x"3", x"3", x"e", x"d", x"f", x"e", x"3", x"1", x"0", x"e", x"0", x"1", x"2", x"0", x"f", x"f", x"1"),
        (x"e", x"0", x"1", x"1", x"2", x"e", x"f", x"1", x"1", x"0", x"0", x"0", x"0", x"2", x"f", x"f", x"0", x"0", x"1", x"f", x"1", x"e", x"0", x"f", x"0", x"f", x"e", x"1", x"f", x"2", x"c", x"e"),
        (x"f", x"e", x"1", x"f", x"1", x"e", x"e", x"f", x"f", x"e", x"0", x"e", x"f", x"1", x"0", x"e", x"e", x"e", x"e", x"0", x"f", x"1", x"f", x"2", x"f", x"1", x"f", x"f", x"1", x"0", x"e", x"e"),
        (x"f", x"1", x"2", x"2", x"0", x"1", x"0", x"e", x"1", x"0", x"0", x"0", x"2", x"e", x"2", x"2", x"0", x"0", x"0", x"e", x"1", x"0", x"1", x"0", x"e", x"0", x"2", x"e", x"f", x"e", x"0", x"f")
    );    
    signal l2b : layer_bias_vector_type(0 to l2o_size - 1)(dec_bw + frc_bw - 1 + 1 downto 0) :=( -- +1 from sign bit.
        x"0", x"2", x"f", x"1", x"f", x"f", x"1", x"e", x"0", x"1", x"0", x"1", x"f", x"1", x"e", x"f", x"f", x"0", x"0", x"1", x"0", x"0", x"1", x"f", x"1", x"f", x"0", x"0", x"1", x"1", x"0", x"e"
    );
    signal l3w : layer_weight_mtx_type(0 to l3i_size - 1, 0 to l3o_size - 1)(dec_bw + frc_bw - 1 + 1 downto 0) :=( -- +1 from sign bit.
        (x"0", x"1", x"1", x"f", x"f", x"0", x"f", x"0", x"f", x"0", x"f", x"f", x"0", x"f", x"f", x"f", x"0", x"f", x"0", x"0"),
        (x"1", x"f", x"1", x"f", x"1", x"0", x"e", x"0", x"f", x"1", x"0", x"1", x"0", x"0", x"f", x"f", x"1", x"f", x"1", x"e"),
        (x"1", x"0", x"f", x"f", x"0", x"0", x"e", x"1", x"e", x"e", x"0", x"0", x"2", x"e", x"1", x"0", x"1", x"1", x"0", x"1"),
        (x"f", x"f", x"0", x"e", x"1", x"1", x"1", x"1", x"e", x"0", x"0", x"f", x"0", x"e", x"e", x"1", x"e", x"f", x"1", x"e"),
        (x"1", x"f", x"f", x"0", x"1", x"f", x"0", x"e", x"0", x"f", x"0", x"0", x"d", x"0", x"f", x"f", x"e", x"0", x"1", x"e"),
        (x"4", x"2", x"1", x"0", x"2", x"1", x"0", x"1", x"f", x"e", x"1", x"2", x"2", x"0", x"0", x"1", x"f", x"1", x"3", x"0"),
        (x"2", x"3", x"f", x"f", x"2", x"2", x"0", x"2", x"e", x"0", x"2", x"3", x"3", x"f", x"f", x"1", x"f", x"1", x"1", x"1"),
        (x"b", x"b", x"e", x"0", x"c", x"b", x"e", x"c", x"f", x"f", x"d", x"d", x"b", x"e", x"f", x"c", x"1", x"d", x"e", x"3"),
        (x"2", x"1", x"0", x"1", x"0", x"1", x"f", x"1", x"e", x"0", x"2", x"f", x"3", x"f", x"0", x"1", x"f", x"e", x"f", x"1"),
        (x"f", x"f", x"0", x"0", x"f", x"f", x"1", x"f", x"e", x"e", x"0", x"1", x"0", x"0", x"e", x"0", x"f", x"0", x"f", x"0"),
        (x"0", x"0", x"e", x"f", x"0", x"1", x"f", x"f", x"0", x"e", x"f", x"0", x"0", x"e", x"0", x"0", x"f", x"0", x"0", x"f"),
        (x"0", x"1", x"1", x"f", x"f", x"f", x"1", x"1", x"0", x"f", x"0", x"f", x"1", x"e", x"e", x"f", x"f", x"f", x"f", x"f"),
        (x"f", x"0", x"0", x"f", x"0", x"1", x"0", x"1", x"f", x"f", x"1", x"1", x"2", x"0", x"0", x"0", x"0", x"f", x"1", x"e"),
        (x"0", x"0", x"e", x"f", x"0", x"1", x"0", x"f", x"f", x"0", x"f", x"1", x"1", x"f", x"e", x"f", x"e", x"e", x"1", x"f"),
        (x"0", x"0", x"f", x"f", x"2", x"1", x"f", x"1", x"0", x"0", x"0", x"0", x"3", x"e", x"0", x"2", x"f", x"f", x"2", x"2"),
        (x"1", x"2", x"1", x"f", x"0", x"0", x"0", x"0", x"f", x"1", x"1", x"0", x"2", x"f", x"e", x"f", x"e", x"f", x"1", x"0"),
        (x"1", x"1", x"e", x"0", x"1", x"2", x"e", x"2", x"e", x"0", x"1", x"2", x"2", x"f", x"e", x"0", x"1", x"1", x"0", x"1"),
        (x"0", x"f", x"f", x"f", x"f", x"f", x"f", x"0", x"0", x"f", x"e", x"0", x"e", x"e", x"0", x"f", x"0", x"f", x"f", x"e"),
        (x"1", x"5", x"0", x"1", x"f", x"0", x"0", x"7", x"1", x"0", x"f", x"1", x"5", x"0", x"1", x"0", x"f", x"0", x"0", x"3"),
        (x"1", x"f", x"e", x"f", x"0", x"f", x"0", x"e", x"e", x"f", x"0", x"f", x"0", x"0", x"e", x"1", x"0", x"1", x"0", x"1"),
        (x"f", x"f", x"0", x"0", x"0", x"0", x"0", x"f", x"e", x"0", x"f", x"f", x"e", x"e", x"f", x"1", x"0", x"0", x"1", x"e"),
        (x"2", x"1", x"0", x"1", x"0", x"1", x"1", x"2", x"0", x"f", x"2", x"2", x"3", x"f", x"1", x"0", x"0", x"f", x"0", x"f"),
        (x"0", x"1", x"0", x"1", x"0", x"0", x"1", x"1", x"0", x"f", x"1", x"1", x"1", x"e", x"f", x"0", x"f", x"1", x"0", x"1"),
        (x"f", x"f", x"e", x"f", x"0", x"0", x"0", x"1", x"e", x"f", x"0", x"1", x"f", x"f", x"0", x"1", x"0", x"1", x"1", x"f"),
        (x"e", x"0", x"1", x"0", x"f", x"e", x"0", x"0", x"0", x"f", x"f", x"f", x"e", x"0", x"e", x"0", x"f", x"0", x"f", x"2"),
        (x"f", x"1", x"f", x"0", x"1", x"0", x"f", x"f", x"0", x"e", x"f", x"f", x"e", x"f", x"f", x"1", x"0", x"d", x"f", x"0"),
        (x"2", x"1", x"1", x"f", x"0", x"0", x"e", x"1", x"0", x"f", x"0", x"0", x"2", x"f", x"f", x"0", x"f", x"1", x"1", x"0"),
        (x"0", x"0", x"e", x"f", x"1", x"0", x"f", x"f", x"0", x"e", x"1", x"f", x"1", x"0", x"1", x"f", x"0", x"0", x"1", x"f"),
        (x"0", x"0", x"e", x"0", x"1", x"1", x"f", x"1", x"1", x"f", x"f", x"1", x"1", x"f", x"0", x"1", x"0", x"f", x"0", x"e"),
        (x"0", x"f", x"0", x"f", x"f", x"1", x"f", x"f", x"0", x"0", x"f", x"1", x"e", x"1", x"e", x"f", x"f", x"0", x"f", x"0"),
        (x"2", x"0", x"0", x"0", x"2", x"0", x"0", x"1", x"f", x"f", x"0", x"2", x"3", x"f", x"0", x"2", x"2", x"3", x"0", x"f"),
        (x"1", x"f", x"0", x"0", x"0", x"f", x"0", x"0", x"0", x"f", x"0", x"f", x"0", x"f", x"f", x"f", x"1", x"e", x"e", x"f")
    );    
    signal l3b : layer_bias_vector_type(0 to l3o_size - 1)(dec_bw + frc_bw - 1 + 1 downto 0) :=( -- +1 from sign bit.
        x"0", x"f", x"f", x"e", x"1", x"0", x"f", x"f", x"0", x"f", x"0", x"0", x"e", x"0", x"0", x"1", x"f", x"f", x"1", x"e"
    );
    -- 0 => due to super-weird VHDL rule, see https://stackoverflow.com/questions/35359413/2d-unconstrained-nx1-array/35362198#35362198
    signal l4w : layer_weight_mtx_type(0 to l4i_size - 1, 0 to l4o_size - 1)(dec_bw + frc_bw - 1 + 1 downto 0) :=( -- +1 from sign bit.
        (0 => x"0"),
        (0 => x"2"),
        (0 => x"f"),
        (0 => x"1"),
        (0 => x"2"),
        (0 => x"2"),
        (0 => x"f"),
        (0 => x"2"),
        (0 => x"f"),
        (0 => x"f"),
        (0 => x"1"),
        (0 => x"0"),
        (0 => x"4"),
        (0 => x"3"),
        (0 => x"e"),
        (0 => x"1"),
        (0 => x"1"),
        (0 => x"2"),
        (0 => x"1"),
        (0 => x"c")
    );    
    signal l4b : layer_bias_vector_type(0 to l4o_size - 1)(dec_bw + frc_bw - 1 + 1 downto 0) :=( -- +1 from sign bit.
        0 => x"e"
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
