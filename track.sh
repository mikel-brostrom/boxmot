source='0'
output='runs/track/exp'
track_config='track_configs/default.yml'
out_config='out_configs/default.yml'

python track.py --source $source --output $output --track_config $track_config --out_config $out_config
          