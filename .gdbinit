b tensorflow::dbg_breakpoint

# define str
#     call (void)printf("> gdb: str $arg0\n")
#     call $arg0.Print(std::cout, 1)
#     call (void)printf("\n")
#     # Flush anything that Print wrote to std::cout.
#     call fflush(0)
# end
