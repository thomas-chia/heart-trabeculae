python3 -m apex3d.run --surface mesh --mode kinematic --hdot 0.008
python3 -m apex3d.run --surface mesh --mode kinematic --hdot 0.08
python3 -m apex3d.run --surface mesh --mode fsi --sigma 2
python3 -m apex3d.run --surface mesh --mode fsi --sigma 0.5
python3 -m apex3d.run --surface mesh --mode kinematic --hdot 0.025
python3 -m apex3d.run --surface mesh --mode kinematic --hdot 0.025 --dx 0.15e-3 --tag _dx150
python3 -m apex3d.run --surface mesh --mode kinematic --hdot 0.025 --dx 0.2e-3 --tag _dx200
