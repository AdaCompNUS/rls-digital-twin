#!/usr/bin/env python

import json

file = 'known_locations.txt'
with open(file, "r") as json_file:
	known_locations = json.load(json_file)

	try:
		location_name = raw_input("Enter a location_name: ")
		location_pos_x = int(raw_input("Enter x coordinate: "))
		location_pos_y = int(raw_input("Enter y coordinate: "))

		known_locations[location_name] = {}
		known_locations[location_name]['x'] = location_pos_x
		known_locations[location_name]['y'] = location_pos_y

		with open(file, "w") as json_file:
			json.dump(known_locations, json_file)
			json_file.close()
	except Exception as e:
		print(e)


