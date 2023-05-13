---
layout: post
title:  Windows VM with VIFO passthrough
date:   2023-05-07 19:24:45 -0700
categories: linux
---

## Credits
To install Windows with VIFO passthrough, I followed this [guide](https://github.com/QaidVoid/Complete-Single-GPU-Passthrough#video-card-driver-virtualisation-detection).
I had to follow some extra steps to make it work on my computer.
You should read and use the original guide first and look back here if you have the same problems I had with the original guide.

## System Setup

Make sure that your BIOS supports virtualization, this should be either _Intel VT-d_ or _AMD-Vi_ in BIOS settings.
Next, add the following kernel parameters `... intel_iommu=on iommu=pt ...` or `... amd_iommu=on iommu=pt ...`.
For GRUB users, this is found in `/etc/default/grub` and for REFIND users this was found for me at `/boot/refind_linux.conf`.
GRUB users should then regenerate their grub config with `grub-mkconfig -o /boot/grub/grub.cfg`.

Next, run the following script:

```sh
#!/bin/bash
shopt -s nullglob
for g in `find /sys/kernel/iommu_groups/* -maxdepth 0 -type d | sort -V`; do
    echo "IOMMU Group ${g##*/}:"
    for d in $g/devices/*; do
        echo -e "\t$(lspci -nns ${d##*/})"
    done;
done;
```

Ideally, this will show everything with your GPU in one single IOMMU group like below:
```
IOMMU Group 21:
       0a:00.1 Audio device [0403]: NVIDIA Corporation AD102 High Definition Audio Controller [10de:22ba] (rev a1)
       0a:00.0 VGA compatible controller [0300]: NVIDIA Corporation AD102 [GeForce RTX 4090] [10de:2684] (rev a1)
```

If the IOMMU group contains slightly more things it should be fine, but make sure you read exactly what is there.
When doing PCI passthrough, you *NEED* to include everything from the IOMMU groups that your card is in.
For me, this IOMMU group originally contained everything, including my NVME disk and etherenet controller.
If this is the case you should follow the instructions in *ACS Override Patch* (hopefully not because it can be a pain).

### ACS Override Patch

For ACS Override Patch, you must get a kernel that has it.
You can either build this kernel yourself after applying the patch or install the Zen Kernel.
For Arch Linux this can be done with `sudo pacman -S linux-zen-headers linux-zen nvidia-dkms`.
Make sure you install `nvidia-dkms` because the standard `nvidia` package will not work with the Zen Kernel.
Next, make sure Zen is added to either your GRUB or REFIND.
For me, I had to edit my REFIND config to ensure that Zen had an `initrd` which was not set for some reason.

After installing Zen and adding it to GRUB or REFIND, add `pcie_acs_override=downstream,multifunction` to your kernel options.

## Making the VM

To make and manage virtual machines, I like to use `virt-manager`.
It provides a nice GUI for `libvirt` which runs on a `qemu` backend.

To start off, download a Windows 10 iso from Microsoft, this [link](https://www.microsoft.com/en-us/software-download/windows10ISO) should work.
Then download the [virtio](https://fedorapeople.org/groups/virt/virtio-win/direct-downloads/stable-virtio/virtio-win.iso) drivers.
Without this, you will not be able to install Windows properly.
Now, to install Windows, launch `virt-manager` and proceed as normal until clicking *Customize before install* on the final step.

We need to do the following for this to work (following is copied from the guide):
1. In _Overview_ section, set _Chipset_ to _Q35_, and _Firmware_ to _UEFI_
2. In _CPUs_ section, set _CPU_ model to _host-passthrough_, and _CPU Topology_ to whatever fits your system.
3. For _SATA_ disk of VM, set _Disk Bus_ to _virtio_.
4. In _NIC_ section, set _Device Model_ to _virtio_
5. _Add Hardware_ > CDROM: virtio-win.iso

When doing this, I found a problem with step *2*.
I set _CPU Topology_ to the option to _Copy host CPU configuration_ and did not manually set the topology myself.
After intalling Windows, this resulted in a single core/thread for the VM.
To fix this, I had to manually set my topology.
To do this, set _sockets_ to 1 (unless you have multiple CPUs in your system?), _cores_ to the number of cores your CPU has, and _threads_ to 2 (assuming 2 threads per core).
To double check if this works, make sure the _vCPU_ allocation matches the number of threads on your system.

### PCI Devices

I suggest installing Windows normally from this point on before continuing with the VIFO passthrough.

Once Windows is installed, remove Channel Spice, Display Spice, Video QXL, Sound ich* and other unnecessary devices.
Now click on _Add Hardware_ and add PCI Devices which match everything from your GPU.

Follow the instructions [here](https://github.com/QaidVoid/Complete-Single-GPU-Passthrough#libvirt-hooks) to add libvirt hooks.

### Keyboard/Mouse Passthrough

While the guide does offer two methods, using either _USB Host Device_ option from virtmanager or _Edev passthrough_, I found only the first method worked.
Fortunetly, this is the easier of the two methods, all you need is to go to _Add Hardware_ > _USB Host Device_ and add your keyboard and mouse.
