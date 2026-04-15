"""Vehicle spawn event definitions for the AgentEvac simulation.

``SPAWN_EVENTS`` is a list of tuples, each defining one agent vehicle to be inserted
into the SUMO simulation.  The main loop (``agentevac.simulation.main``)
iterates this list each decision tick and spawns vehicles whose departure time
has been reached.

Tuple format:
    (veh_id, spawn_edge, dest_edge, depart_time_s, lane, pos, speed, color)

    - ``veh_id``       : Unique SUMO vehicle ID (also used as the agent state key).
    - ``spawn_edge``   : SUMO edge ID where the vehicle is inserted.
    - ``dest_edge``    : Initial destination edge (may be overridden by the LLM).
    - ``depart_time_s``: Simulation time (seconds) at which the vehicle becomes eligible
                         for departure evaluation.
    - ``lane``         : SUMO departure lane specifier (e.g., ``"first"``).
    - ``pos``          : Departure position on the edge in metres.
    - ``speed``        : Departure speed (``"max"`` uses the lane speed limit).
    - ``color``        : RGBA color tuple used for the initial SUMO vehicle color.

Active groups (veh1-veh5): 12 vehicles across 5 spawn locations; the baseline
scenario for development and testing.

Commented-out groups (veh6-veh47): Additional spawn locations across the road network.
Disabled to keep the active agent count manageable.  Re-enable individual groups
to test denser evacuation scenarios.
"""

C_RED = (255, 0, 0, 255)
C_ORANGE = (255, 125, 0, 255)
C_YELLOW = (255, 255, 0, 255)
C_SPRING = (125, 255, 0, 255)
C_GREEN = (0, 255, 0, 255)
C_CYAN = (0, 255, 255, 255)
C_OCEAN = (0, 125, 255, 255)
C_BLUE = (0, 0, 255, 255)
C_VIOLET = (125, 0, 255, 255)
C_MAGENTA = (255, 0, 255, 255)

SPAWN_EVENTS = [
    # vehicle id, spawn edge, dest edge (initial), depart time, lane, pos, speed, (color)
    ("veh1_1", "42006672", "E#S2", 0.0, "first", "10", "max", C_RED),
    ("veh1_2", "42006672", "E#S2", 5.0, "first", "10", "max", C_BLUE),
    ("veh1_3", "42006672", "E#S2", 10.0, "first", "10", "max", C_GREEN),

    ("veh2_1", "42006514#4", "E#S2", 0.0, "first", "20", "max", C_RED),
    ("veh2_2", "42006514#4", "E#S2", 5.0, "first", "20", "max", C_BLUE),
    #
    ("veh3_1", "-42006515", "E#S2", 0.0, "first", "20", "max", C_RED),
    ("veh3_2", "-42006515", "E#S2", 5.0, "first", "20", "max", C_BLUE),

    ("veh4_1", "42006515", "E#S2", 0.0, "first", "20", "max", C_RED),
    ("veh4_2", "42006515", "E#S2", 5.0, "first", "20", "max", C_BLUE),

    ("veh5_1", "42006565", "E#S2", 0.0, "first", "20", "max", C_RED),
    ("veh5_2", "42006565", "E#S2", 5.0, "first", "20", "max", C_BLUE),

    ("veh6_1", "-42006513#0", "E#S2", 0.0, "first", "20", "max", C_RED),
    ("veh6_2", "-42006513#0", "E#S2", 5.0, "first", "20", "max", C_BLUE),
    ("veh6_3", "-42006513#0", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    ("veh6_4", "-42006513#0", "E#S2", 15.0, "first", "20", "max", C_ORANGE),
    ("veh6_5", "-42006513#0", "E#S2", 20.0, "first", "20", "max", C_SPRING),

    ("veh7_1", "42006504#1", "E#S2", 0.0, "first", "20", "max", C_RED),
    ("veh7_2", "42006504#1", "E#S2", 5.0, "first", "20", "max", C_BLUE),
    ("veh7_3", "42006504#1", "E#S2", 10.0, "first", "20", "max", C_GREEN),

    ("veh8_1", "42006513#0", "E#S2", 0.0, "first", "20", "max", C_RED),
    ("veh8_2", "42006513#0", "E#S2", 5.0, "first", "20", "max", C_BLUE),
    ("veh8_3", "42006513#0", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    ("veh8_4", "42006513#0", "E#S2", 15.0, "first", "20", "max", C_ORANGE),
    ("veh8_5", "42006513#0", "E#S2", 20.0, "first", "20", "max", C_SPRING),
    #
    ("veh9_1", "-42006719#1", "E#S2", 0.0, "first", "20", "max", C_RED),
    ("veh9_2", "-42006719#1", "E#S2", 5.0, "first", "20", "max", C_BLUE),
    ("veh9_3", "-42006719#1", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    ("veh9_4", "-42006719#1", "E#S2", 15.0, "first", "20", "max", C_ORANGE),
    ("veh9_5", "-42006719#1", "E#S2", 20.0, "first", "20", "max", C_SPRING),
    ("veh9_6", "-42006719#1", "E#S2", 25.0, "first", "20", "max", C_CYAN),
    ("veh9_7", "-42006719#1", "E#S2", 30.0, "first", "20", "max", C_YELLOW),
    #
    # ("veh10_1", "42006513#1", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh10_2", "42006513#1", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh10_3", "42006513#1", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    # ("veh10_4", "42006513#1", "E#S2", 15.0, "first", "20", "max", C_ORANGE),
    # ("veh10_5", "42006513#1", "E#S2", 20.0, "first", "20", "max", C_SPRING),
    # ("veh10_6", "42006513#1", "E#S2", 25.0, "first", "20", "max", C_CYAN),
    #
    # ("veh11_1", "-42006513#2", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh11_2", "-42006513#2", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh11_3", "-42006513#2", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    # ("veh11_4", "-42006513#2", "E#S2", 15.0, "first", "20", "max", C_ORANGE),
    # ("veh11_5", "-42006513#2", "E#S2", 20.0, "first", "20", "max", C_SPRING),
    # ("veh11_6", "-42006513#2", "E#S2", 25.0, "first", "20", "max", C_CYAN),
    # ("veh11_7", "-42006513#2", "E#S2", 30.0, "first", "20", "max", C_OCEAN),
    # ("veh11_8", "-42006513#2", "E#S2", 35.0, "first", "20", "max", C_VIOLET),
    # ("veh11_9", "-42006513#2", "E#S2", 40.0, "first", "20", "max", C_MAGENTA),
    #
    # ("veh12_1", "30689314#5", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh12_2", "30689314#5", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh12_3", "30689314#5", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    # ("veh12_4", "30689314#5", "E#S2", 15.0, "first", "20", "max", C_ORANGE),
    #
    # ("veh13_1", "-30689314#5", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh13_2", "-30689314#5", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh13_3", "-30689314#5", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    # ("veh13_4", "-30689314#5", "E#S2", 15.0, "first", "20", "max", C_ORANGE),
    # ("veh13_5", "-30689314#5", "E#S2", 20.0, "first", "20", "max", C_SPRING),
    # ("veh13_6", "-30689314#5", "E#S2", 25.0, "first", "20", "max", C_CYAN),
    #
    # ("veh14_1", "42006513#2", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh14_2", "42006513#2", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh14_3", "42006513#2", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    # ("veh14_4", "42006513#2", "E#S2", 15.0, "first", "20", "max", C_ORANGE),
    # ("veh14_5", "42006513#2", "E#S2", 20.0, "first", "20", "max", C_SPRING),
    # ("veh14_6", "42006513#2", "E#S2", 25.0, "first", "20", "max", C_CYAN),
    # ("veh14_7", "42006513#2", "E#S2", 30.0, "first", "20", "max", C_OCEAN),
    # ("veh14_8", "42006513#2", "E#S2", 35.0, "first", "20", "max", C_VIOLET),
    # ("veh14_9", "42006513#2", "E#S2", 40.0, "first", "20", "max", C_MAGENTA),
    #
    # ("veh15_1", "-30689314#4", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh15_2", "-30689314#4", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh15_3", "-30689314#4", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    # ("veh15_4", "-30689314#4", "E#S2", 15.0, "first", "20", "max", C_ORANGE),
    # ("veh15_5", "-30689314#4", "E#S2", 20.0, "first", "20", "max", C_SPRING),
    # ("veh15_6", "-30689314#4", "E#S2", 25.0, "first", "20", "max", C_CYAN),
    # ("veh15_7", "-30689314#4", "E#S2", 30.0, "first", "20", "max", C_OCEAN),
    #
    # ("veh16_1", "-42006513#3", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh16_2", "-42006513#3", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh16_3", "-42006513#3", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    # ("veh16_4", "-42006513#3", "E#S2", 15.0, "first", "20", "max", C_ORANGE),
    # ("veh16_5", "-42006513#3", "E#S2", 20.0, "first", "20", "max", C_SPRING),
    #
    # ("veh17_1", "42006513#3", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh17_2", "42006513#3", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh17_3", "42006513#3", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    #
    # ("veh18_1", "42006734#0", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh18_2", "42006734#0", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh18_3", "42006734#0", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    # ("veh18_4", "42006734#0", "E#S2", 15.0, "first", "20", "max", C_ORANGE),
    # ("veh18_5", "42006734#0", "E#S2", 20.0, "first", "20", "max", C_SPRING),
    # ("veh18_6", "42006734#0", "E#S2", 25.0, "first", "20", "max", C_CYAN),
    # ("veh18_7", "42006734#0", "E#S2", 30.0, "first", "20", "max", C_OCEAN),
    # ("veh18_8", "42006734#0", "E#S2", 35.0, "first", "20", "max", C_VIOLET),
    #
    # ("veh19_1", "-42006513#4", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh19_2", "-42006513#4", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh19_3", "-42006513#4", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    #
    # ("veh20_1", "42006513#4", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh20_2", "42006513#4", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    #
    # ("veh21_1", "30689314#0", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh21_2", "30689314#0", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh21_3", "30689314#0", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    # ("veh21_4", "30689314#0", "E#S2", 15.0, "first", "20", "max", C_ORANGE),
    #
    # ("veh22_1", "-30689314#0", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh22_2", "-30689314#0", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh22_3", "-30689314#0", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    # ("veh22_4", "-30689314#0", "E#S2", 15.0, "first", "20", "max", C_ORANGE),
    #
    # ("veh23_1", "42006734#1", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh23_2", "42006734#1", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh23_3", "42006734#1", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    # ("veh23_4", "42006734#1", "E#S2", 15.0, "first", "20", "max", C_ORANGE),
    #
    # ("veh24_1", "42006713#1", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh24_2", "42006713#1", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh24_3", "42006713#1", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    #
    # ("veh25_1", "42006701#0", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh25_2", "42006701#0", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh25_3", "42006701#0", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    # ("veh25_4", "42006701#0", "E#S2", 15.0, "first", "20", "max", C_ORANGE),
    # ("veh25_5", "42006701#0", "E#S2", 20.0, "first", "20", "max", C_SPRING),
    #
    # ("veh26_1", "479505716#1", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh26_2", "479505716#1", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh26_3", "479505716#1", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    # ("veh26_4", "479505716#1", "E#S2", 15.0, "first", "20", "max", C_ORANGE),
    #
    # ("veh27_1", "-479505716#1", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh27_2", "-479505716#1", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh27_3", "-479505716#1", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    # ("veh27_4", "-479505716#1", "E#S2", 15.0, "first", "20", "max", C_ORANGE),
    #
    # ("veh28_1", "42006734#2", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh28_2", "42006734#2", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh28_3", "42006734#2", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    #
    # ("veh29_1", "42006734#2", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh29_2", "42006734#2", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    #
    # ("veh30_1", "-42006522#1", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh30_2", "-42006522#1", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh30_3", "-42006522#1", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    #
    # ("veh31_1", "42006522#1", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh31_2", "42006522#1", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    #
    # ("veh32_1", "42006636#0", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh32_2", "42006636#0", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh32_3", "42006636#0", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    # ("veh32_4", "42006636#0", "E#S2", 15.0, "first", "20", "max", C_ORANGE),
    # ("veh32_5", "42006636#0", "E#S2", 20.0, "first", "20", "max", C_SPRING),
    #
    # ("veh33_1", "-966804140", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh33_2", "-966804140", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh33_3", "-966804140", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    #
    # ("veh34_1", "42006708", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh34_2", "42006708", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh34_3", "42006708", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    #
    # ("veh35_1", "479505354#2", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh35_2", "479505354#2", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh35_3", "479505354#2", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    #
    # ("veh36_1", "-42006660", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh36_2", "-42006660", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh36_3", "-42006660", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    # ("veh36_4", "-42006660", "E#S2", 15.0, "first", "20", "max", C_ORANGE),
    # ("veh36_5", "-42006660", "E#S2", 20.0, "first", "20", "max", C_SPRING),
    #
    # ("veh37_1", "42006589", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh37_2", "42006589", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh37_3", "42006589", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    #
    # ("veh38_1", "42006572", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh38_2", "42006572", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    #
    # ("veh39_1", "42006733", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh39_2", "42006733", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh39_3", "42006733", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    #
    # ("veh40_1", "42006506", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh40_2", "42006506", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh40_3", "42006506", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    # ("veh40_4", "42006506", "E#S2", 15.0, "first", "20", "max", C_ORANGE),
    # ("veh40_5", "42006506", "E#S2", 20.0, "first", "20", "max", C_SPRING),
    #
    # ("veh41_1", "-42006549#1", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh41_2", "-42006549#1", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh41_3", "-42006549#1", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    # ("veh41_4", "-42006549#1", "E#S2", 15.0, "first", "20", "max", C_ORANGE),
    # ("veh41_5", "-42006549#1", "E#S2", 20.0, "first", "20", "max", C_SPRING),
    # ("veh41_6", "-42006549#1", "E#S2", 25.0, "first", "20", "max", C_CYAN),
    # ("veh41_7", "-42006549#1", "E#S2", 30.0, "first", "20", "max", C_OCEAN),
    # ("veh41_8", "-42006549#1", "E#S2", 35.0, "first", "20", "max", C_VIOLET),
    #
    # ("veh42_1", "-42006552#1", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh42_2", "-42006552#1", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh42_3", "-42006552#1", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    # ("veh42_4", "-42006552#1", "E#S2", 15.0, "first", "20", "max", C_ORANGE),
    # ("veh42_5", "-42006552#1", "E#S2", 20.0, "first", "20", "max", C_SPRING),
    #
    # ("veh43_1", "42006552#0", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh43_2", "42006552#0", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh43_3", "42006552#0", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    # ("veh43_4", "42006552#0", "E#S2", 15.0, "first", "20", "max", C_ORANGE),
    # ("veh43_5", "42006552#0", "E#S2", 20.0, "first", "20", "max", C_SPRING),
    #
    # ("veh44_1", "-42006552#0", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh44_2", "-42006552#0", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh44_3", "-42006552#0", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    # ("veh44_4", "-42006552#0", "E#S2", 15.0, "first", "20", "max", C_ORANGE),
    # ("veh44_5", "-42006552#0", "E#S2", 20.0, "first", "20", "max", C_SPRING),
    #
    # ("veh45_1", "-42006706#0", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh45_2", "-42006706#0", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh45_3", "-42006706#0", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    # ("veh45_4", "-42006706#0", "E#S2", 15.0, "first", "20", "max", C_ORANGE),
    # ("veh45_5", "-42006706#0", "E#S2", 20.0, "first", "20", "max", C_SPRING),
    #
    # ("veh46_1", "42006706#1", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh46_2", "42006706#1", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
    # ("veh46_3", "42006706#1", "E#S2", 10.0, "first", "20", "max", C_GREEN),
    # ("veh46_4", "42006706#1", "E#S2", 15.0, "first", "20", "max", C_ORANGE),
    # ("veh46_5", "42006706#1", "E#S2", 20.0, "first", "20", "max", C_SPRING),
    # ("veh46_6", "42006706#1", "E#S2", 25.0, "first", "20", "max", C_CYAN),
    # ("veh46_7", "42006706#1", "E#S2", 30.0, "first", "20", "max", C_OCEAN),
    #
    # ("veh47_1", "42006592", "E#S2", 0.0, "first", "20", "max", C_RED),
    # ("veh47_2", "42006592", "E#S2", 5.0, "first", "20", "max", C_YELLOW),
]
