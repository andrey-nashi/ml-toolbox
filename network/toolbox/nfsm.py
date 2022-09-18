import os
import json
import getpass
import subprocess

class NetworkFileSystemManager:

    NETWORK_MAP_FILE = "network-map.json"

    KEY_NETWORK_MAP = "network_map"
    KEY_SSH_LOCATION_MAP = "ssh_location_map"
    KEY_SSH_KEY_FILE = "ssh_key"


    def __init__(self):
        self.path_resources = os.path.abspath(os.path.dirname( __file__ ))
        self.path_resources = os.path.join(self.path_resources, "resources")

        f = open(os.path.join(self.path_resources, self.NETWORK_MAP_FILE))
        data = json.load(f)
        f.close()

        self.network_map = data[self.KEY_NETWORK_MAP]
        self.sshfs_map = data[self.KEY_SSH_LOCATION_MAP]
        self.ssh_key = data[self.KEY_SSH_KEY_FILE]

        if self.ssh_key is not None:
            self.path_key = os.path.join(self.path_resources, self.ssh_key)

    def execute_mount(self, sshfs_name, user=None):
        if user is None:
            user = getpass.getuser()

        if sshfs_name not in self.sshfs_map:
            print("[ERROR]: No location", sshfs_name)

        path_server = self.sshfs_map[sshfs_name][0]
        path_local = self.sshfs_map[sshfs_name][1]
        server_name = self.sshfs_map[sshfs_name][2]
        server_ip = self.network_map[server_name]

        command = "sshfs -o IdentityFile=" + self.path_key + " "
        command += user +"@" + server_ip + ":" + path_server + " " + path_local + " "
        command += "-o reconnect,transform_symlinks,allow_other"

        os.system(command)

    def execute_ssh(self, machine_name, user=None):
        if user is None:
            user = getpass.getuser()

        if machine_name not in self.network_map:
            print("[ERROR]: No machine with name", machine_name)

        server_ip = self.network_map[machine_name]

        command = "ssh -i " + self.path_key + " "
        command += user + "@" + server_ip
        os.system(command)

    def get_info_machines(self):
        for machine in self.network_map:
            print(machine, self.network_map[machine])

    def get_info_mountpoints(self):
        for mountpoint in self.sshfs_map:
            print(mountpoint, self.sshfs_map[mountpoint])
