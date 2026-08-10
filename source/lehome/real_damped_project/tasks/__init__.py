"""Isaac/LeHome environment access, retained for evaluating policies in sim.

The custom training stack (mock backend, reward shaping, task env, parallel
envs) was retired when the project moved to finetuning a pretrained VLA. What
remains is what a policy still needs to be *scored*: a live LeHome garment
environment and a guarded Isaac launcher.
"""
