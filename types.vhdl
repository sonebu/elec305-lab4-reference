library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

package nn_types is
    type layer_weight_mtx_type is array(integer range <>, integer range <>) of std_logic_vector;
    type layer_bias_vector_type is array(integer range <>) of std_logic_vector;
    type layer_io_vector_type is array(integer range <>) of std_logic_vector;
end package nn_types;