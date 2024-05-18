library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

-- for nearest pow of 2 (non-synth)
use IEEE.MATH_REAL.all;
use IEEE.math_real."ceil";
use IEEE.math_real."log2";

use work.nn_types.all;

entity layer is
    Generic (INPUT_SIZE     : integer := 8;
             OUTPUT_SIZE    : integer := 16;
             HAS_RELU       : std_logic := '0'; 
             DEC_BITWIDTH   : integer := 0; -- this is the bitwidth that the original input, weights and biases use
             CARRY_DEC_BW   : integer := 0; -- this is the bitwidth that gets carried over from the previous layer due to expansion on the decimal part of the Q format
             FRC_BITWIDTH   : integer := 15);
    Port (clk       : in STD_LOGIC;
          ena       : in STD_LOGIC;
          done      : out STD_LOGIC;
          weights_i : in layer_weight_mtx_type(0 to INPUT_SIZE - 1, 0 to OUTPUT_SIZE - 1)(DEC_BITWIDTH + FRC_BITWIDTH + 1 - 1 downto 0); -- +1 from sign bit.
          biases_i  : in layer_bias_vector_type(0 to OUTPUT_SIZE - 1)(DEC_BITWIDTH + FRC_BITWIDTH + 1 - 1 downto 0); -- +1 from sign bit.
          data_i    : in layer_io_vector_type(0 to INPUT_SIZE - 1)(DEC_BITWIDTH + CARRY_DEC_BW + FRC_BITWIDTH + 1 - 1 downto 0); -- +1 from sign bit.
          data_o    : out layer_io_vector_type(0 to OUTPUT_SIZE - 1)(2*(DEC_BITWIDTH + 1) + CARRY_DEC_BW + FRC_BITWIDTH + 1 + integer(ceil(log2(real(INPUT_SIZE)))) - 1 downto 0)); -- last +1 from sign bit.
end layer;

architecture Behavioral of layer is

    -- DSP units are allowed this time!
    attribute use_dsp : string;
    attribute use_dsp of Behavioral : architecture is "yes";
    
    signal ena_d    : std_logic := '0';
    signal ena_d_re : std_logic := '0';

    signal done_d    : std_logic := '0';
    signal done_d_re : std_logic := '0';
    
    type inputs is array(0 to INPUT_SIZE-1) of signed(DEC_BITWIDTH + CARRY_DEC_BW + FRC_BITWIDTH + 1 - 1 downto 0);
    signal x : inputs := (others=>(others=>'0'));

    type weights is array(0 to INPUT_SIZE - 1, 0 to OUTPUT_SIZE - 1) of signed(DEC_BITWIDTH + FRC_BITWIDTH + 1 - 1 downto 0);
    signal w : weights := (others=>(others=>(others=>'0')));

    type biases is array(0 to OUTPUT_SIZE - 1) of signed(DEC_BITWIDTH + FRC_BITWIDTH + 1 - 1 downto 0);
    signal b : biases := (others=>(others=>'0'));

    -- NOTE: since the weights are *resized* to CARRY_DEC_BW bitwidth, they cannot contain values in those expanded bits, 
    --       therefore we do not count those bits in for the expansion twice (it only leaks in from the input).
    --       If we counted them twice we would have MULT_WIDTH := 2*(DEC_BITWIDTH + FRC_BITWIDTH + CARRY_DEC_BW + 1)
    constant MULT_WIDTH : integer := 2*(DEC_BITWIDTH + FRC_BITWIDTH + 1) + CARRY_DEC_BW; 
    
    constant ACCUM_WIDTH: integer := MULT_WIDTH + integer(ceil(log2(real(INPUT_SIZE))));
    type mults is array(0 to INPUT_SIZE-1, 0 to OUTPUT_SIZE-1) of signed(MULT_WIDTH - 1 downto 0);
    type accums is array(0 to OUTPUT_SIZE-1) of signed(ACCUM_WIDTH - 1 downto 0);
    signal m : mults := (others=>(others=>(others=>'0')));
    signal a : accums := (others=>(others=>'0'));

    constant MAC_WIDTH : integer := ACCUM_WIDTH - FRC_BITWIDTH; -- we'll cut the extra FRC_BITWIDTH off before bias addition
    type macs is array(0 to OUTPUT_SIZE-1) of signed(MAC_WIDTH - 1 downto 0);
    signal c : macs := (others=>(others=>'0'));

    constant Y_WIDTH : integer := MAC_WIDTH + 1; -- note how this is equal to data_o size!
    type outputs is array(0 to OUTPUT_SIZE-1) of signed(Y_WIDTH - 1 downto 0);
    signal y : outputs := (others=>(others=>'0'));
    
    constant zeros_4relu : signed(Y_WIDTH - 1 downto 0) := (others=>'0');

    type layer_state is (s_idle, s_mult, s_accum); -- if we don't do this in two states, the a and c arrays get locked up due to array reduction constraints 
    signal ls_t0, ls_t1: layer_state;
begin
    
    -- Note: I'm not sure if this design will work after implementation with fast clocks since 
    -- the operations within each state are assumed to be finished in 1 cycle here, 
    -- but that might not be the case and we might get invalid outputs at arbitrary times 
    -- if (because?) the state transitions earlier than it should
    --
    -- We probably need something like a synchronization mechanism that triggers a done flag
    -- only when the output gets computed (probably an array diff check?). 
    -- This is a bit advanced though, so let's not include it in this lab.
    
    layer_state_transition: process(clk)
    begin
        if rising_edge(clk) then
            ena_d <= ena;
            done_d <= done;
            ls_t0 <= ls_t1;
        end if;
        ena_d_re <= not ena_d and ena;
        done_d_re <= not done_d and done;
    end process;
    
    layer_state_machine: process(ls_t0, ena, clk)
        variable sumvar : signed(ACCUM_WIDTH - 1 downto 0);
        variable outvar : signed(Y_WIDTH -1 downto 0);
    begin
        for k in 0 to OUTPUT_SIZE-1 loop
            data_o(k) <= std_logic_vector(y(k));
        end loop;
        case ls_t0 is
            when s_idle  =>
                if(ena_d_re = '1') then
                    ls_t1 <= s_mult;
                    done <= '0';
                else
                    ls_t1 <= s_idle;
                    done <= '0';
                end if;
            when s_mult  =>
                for j in 0 to OUTPUT_SIZE-1 loop
                    for i in 0 to INPUT_SIZE-1 loop
                        w(i,j) <= signed(weights_i(i,j));
                        x(i)   <= signed(data_i(i));
                        m(i,j) <= x(i)*w(i,j);
                    end loop;
                end loop;
                done <= '0';
                ls_t1 <= s_accum;
            when s_accum =>
                for j in 0 to OUTPUT_SIZE-1 loop
                    sumvar := (others => '0');
                    for k in 0 to INPUT_SIZE-1 loop
                         sumvar := sumvar + m(k,j);
                    end loop;
                    a(j) <= sumvar;
                    c(j) <= a(j)(ACCUM_WIDTH - 1 downto FRC_BITWIDTH); -- drop FRC_BITWIDTH number of bits before moving on to bias comp
                    b(j) <= signed(biases_i(j)); 
                    outvar := resize(c(j),Y_WIDTH) + resize(b(j),Y_WIDTH); -- extend the sign bits of b and c to match y
                    if (HAS_RELU = '0') then
                        y(j) <= outvar;
                    else
                        if (outvar > zeros_4relu) then
                            y(j) <= outvar;
                        else
                            y(j) <= zeros_4relu;
                        end if;
                    end if;
                end loop;
                done <= '1';
                ls_t1 <= s_idle;
        end case;
    end process;
    
end Behavioral;
