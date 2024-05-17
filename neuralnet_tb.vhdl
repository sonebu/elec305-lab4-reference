library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

-- for nearest pow of 2 (non-synth)
use IEEE.MATH_REAL.all;
use IEEE.math_real."ceil";
use IEEE.math_real."log2";

use work.nn_types.all;
use std.textio.all;

entity nn_tb is
    Generic (INPUT_FOLDER  : string := "/home/buraksoner/workspace/vivado/lab4_neuralnet_reference/testing_inputs/";
             OUTPUT_FOLDER : string := "/home/buraksoner/workspace/vivado/lab4_neuralnet_reference/testing_outputs/";
             l1i_size : integer := 8;
             l1o_size : integer := 16;
             l2i_size : integer := 16;
             l2o_size : integer := 32;
             l3i_size : integer := 32;
             l3o_size : integer := 20;
             l4i_size : integer := 20;
             l4o_size : integer := 1;
             dec_bw   : integer := 0;
             frc_bw   : integer := 15);
end nn_tb;

architecture Behavioral of nn_tb is
    component neuralnet is
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
              n_data_o : out layer_io_vector_type(0 to l4o_size - 1)(2*(dec_bw+1)+18 + frc_bw + 1 + integer(ceil(log2(real(l4i_size)))) - 1 downto 0)); -- manually computed 18
    end component;
    
    signal clk : std_logic := '0';
    signal ena : std_logic := '0';
    constant clock_period  : time := 10 ns;
    
    signal int_input_s : integer := 0;
    signal data_in : layer_io_vector_type(0 to l1i_size - 1)(dec_bw + frc_bw + 1 - 1 downto 0) := (others=>(others=>'0'));
    signal data_out : layer_io_vector_type(0 to l4o_size - 1)(2*(dec_bw+1)+18 + frc_bw + 1 + integer(ceil(log2(real(l4i_size)))) - 1 downto 0); --  manually computed 18

    -- source: https://stackoverflow.com/a/42557233/3811558    
    function fIntToStringLeading0 (a : natural; d : integer range 1 to 9) return string is
      variable vString : string(1 to d);
    begin
      if(a >= 10**d) then
        return integer'image(a);
      else
        for i in 0 to d-1 loop
          vString(d-i to d-i) := integer'image(a/(10**i) mod 10);
        end loop;
        return vString;
      end if;
    end function;
    
begin
    nn_inst: neuralnet
    generic map (
        l1i_size  => l1i_size,
        l1o_size  => l1o_size,
        l2i_size  => l2i_size,
        l2o_size  => l2o_size,
        l3i_size  => l3i_size,
        l3o_size  => l3o_size,
        l4i_size  => l4i_size,
        l4o_size  => l4o_size,
        dec_bw    => dec_bw,
        frc_bw    => frc_bw
    )
    port map(
        clk    => clk,
        ena    => ena,
        n_data_i => data_in,
        n_data_o => data_out 
    );
    
    process       
        file     input_file  : text;
        variable input_line  : line; 
        file     output_file : text;    
        variable output_line : line;
        variable int_input_v : integer := 0;       
        variable good_v      : boolean;
    begin
        wait for 10 ns;
        for k in 0 to 4127 loop
            file_open(input_file, INPUT_FOLDER & "i_" & fIntToStringLeading0(k, 7) & ".txt",  read_mode);
            file_open(output_file, OUTPUT_FOLDER & "i_" & fIntToStringLeading0(k, 7) & ".txt",  write_mode);
            ena <= '0';
            wait for 10 ns;
            for i in 0 to l1i_size-1 loop
                readline(input_file, input_line);
                read(input_line, int_input_v, good_v);
                int_input_s <= int_input_v;
                wait for 1 ns;
                data_in(i) <= std_logic_vector(to_unsigned(int_input_s, data_in(i)'length));
            end loop;
            wait for 100 ns;
            ena <= '1';
            wait for 200 ns;
            for j in 0 to l4o_size-1 loop
                write(output_line, to_integer(signed(data_out(j))), left, 10);
                writeline(output_file, output_line);
            end loop;    
            wait for 200 ns;
        end loop;
        wait;
    end process;
    
    Clock_gen: process                                
    begin
        clk <= '0', '1' after clock_period/2;
        wait for clock_period; 
    end process;

end Behavioral;
