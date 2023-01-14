# ReachView code is placed under the GPL license.
# Written by Egor Fedorov (egor.fedorov@emlid.com)
# Copyright (c) 2015, Emlid Limited
# All rights reserved.

# If you are interested in using ReachView code as a part of a
# closed source project, please contact Emlid Limited (info@emlid.com).

# This file is part of ReachView.

# ReachView is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# ReachView is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with ReachView.  If not, see <http://www.gnu.org/licenses/>.


## Added some more functionalities by Mehmet Pekmezci (also GPL :) )

##

import time
import pexpect
import subprocess
import sys

class BluetoothctlError(Exception):
    """This exception is raised, when bluetoothctl fails to start."""
    pass


class Bluetoothctl:
    """A wrapper for bluetoothctl utility."""

    def __init__(self):
        out = subprocess.check_output("rfkill unblock bluetooth", shell = True)
        self.child = pexpect.spawn("bluetoothctl", echo = False)

    def get_output(self, command, pause = 0):
        """Run a command in bluetoothctl prompt, return output as a list of lines."""
        self.child.send(command + "\n")
        time.sleep(pause)
        start_failed = self.child.expect(["bluetooth", pexpect.EOF])

        if start_failed:
            raise BluetoothctlError("Bluetoothctl failed after running " + command)

        return self.child.before.decode("utf-8").split("\r\n")

    def start_scan(self):
        """Start bluetooth scanning process."""
        try:
            self.get_output("power off")
            out = self.get_output("scan on")
        except BluetoothctlError as e:
            print(e)
            return None

    def list(self):
        """List bluetooth devices."""
        try:
            out = self.get_output("list")
            for i in out :
                print(i)
        except BluetoothctlError as e:
            print(e)
            return None

    def make_discoverable(self):
        """Make device discoverable."""
        try:
            out = self.get_output("discoverable on")
        except BluetoothctlError as e:
            print(e)
            return None

    def parse_device_info(self, info_string):
        """Parse a string corresponding to a device."""
        device = {}
        block_list = ["[\x1b[0;", "removed"]
        string_valid = not any(keyword in info_string for keyword in block_list)

        if string_valid:
            try:
                device_position = info_string.index("Device")
            except ValueError:
                pass
            else:
                if device_position > -1:
                    attribute_list = info_string[device_position:].split(" ", 2)
                    device = {
                        "mac_address": attribute_list[1],
                        "name": attribute_list[2]
                    }

        return device

    def get_available_devices(self):
        """Return a list of tuples of paired and discoverable devices."""
        try:
            out = self.get_output("devices")
        except BluetoothctlError as e:
            print(e)
            return None
        else:
            available_devices = []
            for line in out:
                device = self.parse_device_info(line)
                if device:
                    available_devices.append(device)

            return available_devices

    def get_paired_devices(self):
        """Return a list of tuples of paired devices."""
        try:
            out = self.get_output("paired-devices")
        except BluetoothctlError as e:
            print(e)
            return None
        else:
            paired_devices = []
            for line in out:
                device = self.parse_device_info(line)
                if device:
                    paired_devices.append(device)

            return paired_devices

    def get_discoverable_devices(self):
        """Filter paired devices out of available."""
        available = self.get_available_devices()
        paired = self.get_paired_devices()

        return [d for d in available if d not in paired]

    def get_device_info(self, mac_address):
        """Get device info by mac address."""
        try:
            out = self.get_output("info " + mac_address)
        except BluetoothctlError as e:
            print(e)
            return None
        else:
            return out

    def pair(self, mac_address):
        """Try to pair with a device by mac address."""
        try:
            out = self.get_output("pair " + mac_address, 4)
        except BluetoothctlError as e:
            print(e)
            return None
        else:
            res = self.child.expect(["Failed to pair", "Pairing successful", pexpect.EOF])
            success = True if res == 1 else False
            return success

    def select(self, mac_address):
        """Try to pair with a device by mac address."""
        try:
            out = self.get_output("select " + mac_address, 4)
        except BluetoothctlError as e:
            print(e)
            return None
        else:
            res = self.child.expect(["Failed to select", "Selecting successful", pexpect.EOF])
            success = True if res == 1 else False
            return success

    def remove(self, mac_address):
        """Remove paired device by mac address, return success of the operation."""
        try:
            out = self.get_output("remove " + mac_address, 3)
        except BluetoothctlError as e:
            print(e)
            return None
        else:
            res = self.child.expect(["not available", "Device has been removed", pexpect.EOF])
            success = True if res == 1 else False
            return success

    def connect(self, mac_address):
        """Try to connect to a device by mac address."""
        try:
            out = self.get_output("connect " + mac_address, 2)
        except BluetoothctlError as e:
            print(e)
            return None
        else:
            res = self.child.expect(["Failed to connect", "Connection successful", pexpect.EOF])
            success = True if res == 1 else False
            return success

    def connect(self, bluetoothReceiverMacAddress, speakerMacAddress):
        """Try to connect to a device by mac address."""
        try:
            self.get_output("select " + bluetoothReceiverMacAddress, 2)
            self.get_output("power off ", 2)
            self.get_output("power on ", 2)
            out = self.get_output("remove " + speakerMacAddress, 2)
            out = self.get_output("connect " + speakerMacAddress, 2)
        except BluetoothctlError as e:
            print(e)
            return None
        else:
            res = self.child.expect(["Failed to connect", "Connection successful", pexpect.EOF])
            success = True if res == 1 else False
            return success

    def disconnect(self, mac_address):
        """Try to disconnect to a device by mac address."""
        try:
            out = self.get_output("disconnect " + mac_address, 2)
        except BluetoothctlError as e:
            print(e)
            return None
        else:
            res = self.child.expect(["Failed to disconnect", "Successful disconnected", pexpect.EOF])
            success = True if res == 1 else False
            return success


if __name__ == "__main__":


#Controller 68:07:15:D4:9E:E5 silisyum #1 [default]
#Controller 00:15:83:0C:BF:EB silisyum #2
#Controller 00:1A:7D:DA:71:10 silisyum
#
#[NEW] Device 3D:45:41:74:43:09 TG-117
#[NEW] Device F9:3D:F7:F1:45:66 TG-117

    print("Init bluetooth...")
    bl = Bluetoothctl()
    print("Ready!")
    bl.list()
    #bl.connect(self,bluetoothReceiverMacAddress, speakerMacAddress)
    bl.connect("00:15:83:0C:BF:EB", "3D:45:41:74:43:09")
    bl.connect("00:1A:7D:DA:71:10", "F9:3D:F7:F1:45:66")

    #bl.start_scan()
    #print("Scanning for 10 seconds...")
    #for i in range(0, 10):
    #    print(i)
    #    time.sleep(1)
#
#    print(bl.get_discoverable_devices())
    #print(bl.get_available_devices())
