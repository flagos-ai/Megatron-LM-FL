"""Keep override dependencies lazy and independent.

Leave implementation imports to the registry so selecting the MoE override
does not also import FSDP, and vice versa. Eager imports here would couple
unrelated core/plugin modules and make circular imports easier to introduce.
"""
