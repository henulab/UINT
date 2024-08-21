#!/usr/bin/env python

from mininet.net import Mininet
from mininet.node import Controller, RemoteController, OVSController
from mininet.node import CPULimitedHost, Host, Node
from mininet.node import OVSKernelSwitch, UserSwitch
from mininet.node import IVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink, Intf
from subprocess import call

def myNetwork():

    net = Mininet( topo=None,
                   build=False,
                   ipBase='10.0.0.0/8')

    info( '*** Adding controller\n' )
    c0=net.addController(name='c0',
                      controller=RemoteController,
                      ip='127.0.0.1',
                      protocol='tcp',
                      port=6633)

    info( '*** Add switches\n')
    s21 = net.addSwitch('s21', cls=OVSKernelSwitch, dpid='0000000000000021')
    s5 = net.addSwitch('s5', cls=OVSKernelSwitch, dpid='0000000000000005')
    s13 = net.addSwitch('s13', cls=OVSKernelSwitch, dpid='0000000000000013')
    s19 = net.addSwitch('s19', cls=OVSKernelSwitch, dpid='0000000000000019')
    s16 = net.addSwitch('s16', cls=OVSKernelSwitch, dpid='0000000000000016')
    s12 = net.addSwitch('s12', cls=OVSKernelSwitch, dpid='0000000000000012')
    s7 = net.addSwitch('s7', cls=OVSKernelSwitch, dpid='0000000000000007')
    s15 = net.addSwitch('s15', cls=OVSKernelSwitch, dpid='0000000000000015')
    s20 = net.addSwitch('s20', cls=OVSKernelSwitch, dpid='0000000000000020')
    s23 = net.addSwitch('s23', cls=OVSKernelSwitch, dpid='0000000000000023')
    s10 = net.addSwitch('s10', cls=OVSKernelSwitch, dpid='0000000000000010')
    s3 = net.addSwitch('s3', cls=OVSKernelSwitch, dpid='0000000000000003')
    s1 = net.addSwitch('s1', cls=OVSKernelSwitch, dpid='0000000000000001')
    s6 = net.addSwitch('s6', cls=OVSKernelSwitch, dpid='0000000000000006')
    s9 = net.addSwitch('s9', cls=OVSKernelSwitch, dpid='0000000000000009')
    s2 = net.addSwitch('s2', cls=OVSKernelSwitch, dpid='0000000000000002')
    s11 = net.addSwitch('s11', cls=OVSKernelSwitch, dpid='0000000000000011')
    s22 = net.addSwitch('s22', cls=OVSKernelSwitch, dpid='0000000000000022')
    s8 = net.addSwitch('s8', cls=OVSKernelSwitch, dpid='0000000000000008')
    s14 = net.addSwitch('s14', cls=OVSKernelSwitch, dpid='0000000000000014')
    s4 = net.addSwitch('s4', cls=OVSKernelSwitch, dpid='0000000000000004')
    s17 = net.addSwitch('s17', cls=OVSKernelSwitch, dpid='0000000000000017')
    s18 = net.addSwitch('s18', cls=OVSKernelSwitch, dpid='0000000000000018')

    info( '*** Add hosts\n')
    h6 = net.addHost('h6', cls=Host, ip='10.0.0.6', defaultRoute=None)
    h3 = net.addHost('h3', cls=Host, ip='10.0.0.3', defaultRoute=None)
    h4 = net.addHost('h4', cls=Host, ip='10.0.0.4', defaultRoute=None)
    h1 = net.addHost('h1', cls=Host, ip='10.0.0.1', defaultRoute=None)
    h5 = net.addHost('h5', cls=Host, ip='10.0.0.5', defaultRoute=None)
    h2 = net.addHost('h2', cls=Host, ip='10.0.0.2', defaultRoute=None)

    info( '*** Add links\n')
    s2s3 = {'bw':1000,'delay':'10'}
    net.addLink(s2, s3, cls=TCLink , **s2s3)
    s3s4 = {'bw':1000,'delay':'10'}
    net.addLink(s3, s4, cls=TCLink , **s3s4)
    s4s5 = {'bw':1000,'delay':'10'}
    net.addLink(s4, s5, cls=TCLink , **s4s5)
    s5h1 = {'bw':10}
    net.addLink(s5, h1, cls=TCLink , **s5h1)
    s1s2 = {'bw':10,'delay':'10'}
    net.addLink(s1, s2, cls=TCLink , **s1s2)
    s1s6 = {'bw':25,'delay':'10'}
    net.addLink(s1, s6, cls=TCLink , **s1s6)
    s1s11 = {'bw':30,'delay':'10'}
    net.addLink(s1, s11, cls=TCLink , **s1s11)
    s1s16 = {'bw':15,'delay':'10'}
    net.addLink(s1, s16, cls=TCLink , **s1s16)
    s1s20 = {'bw':20,'delay':'10'}
    net.addLink(s1, s20, cls=TCLink , **s1s20)
    s1h6 = {'bw':100}
    net.addLink(s1, h6, cls=TCLink , **s1h6)
    s6s7 = {'bw':1000,'delay':'10'}
    net.addLink(s6, s7, cls=TCLink , **s6s7)
    s7s8 = {'bw':1000,'delay':'10'}
    net.addLink(s7, s8, cls=TCLink , **s7s8)
    s8s9 = {'bw':1000,'delay':'10'}
    net.addLink(s8, s9, cls=TCLink , **s8s9)
    s9s10 = {'bw':1000,'delay':'10'}
    net.addLink(s9, s10, cls=TCLink , **s9s10)
    s10h2 = {'bw':25}
    net.addLink(s10, h2, cls=TCLink , **s10h2)
    s11s12 = {'bw':1000,'delay':'10'}
    net.addLink(s11, s12, cls=TCLink , **s11s12)
    s12s13 = {'bw':1000,'delay':'10'}
    net.addLink(s12, s13, cls=TCLink , **s12s13)
    s13s14 = {'bw':1000,'delay':'10'}
    net.addLink(s13, s14, cls=TCLink , **s13s14)
    s14s15 = {'bw':1000,'delay':'10'}
    net.addLink(s14, s15, cls=TCLink , **s14s15)
    s15h3 = {'bw':30}
    net.addLink(s15, h3, cls=TCLink , **s15h3)
    s16s17 = {'bw':1000,'delay':'10'}
    net.addLink(s16, s17, cls=TCLink , **s16s17)
    s17s18 = {'bw':1000,'delay':'10'}
    net.addLink(s17, s18, cls=TCLink , **s17s18)
    s18s19 = {'bw':1000,'delay':'10'}
    net.addLink(s18, s19, cls=TCLink , **s18s19)
    s19h4 = {'bw':15}
    net.addLink(s19, h4, cls=TCLink , **s19h4)
    s20s21 = {'bw':1000,'delay':'10'}
    net.addLink(s20, s21, cls=TCLink , **s20s21)
    s21s22 = {'bw':1000,'delay':'10'}
    net.addLink(s21, s22, cls=TCLink , **s21s22)
    s22s23 = {'bw':1000,'delay':'10'}
    net.addLink(s22, s23, cls=TCLink , **s22s23)
    s23h5 = {'bw':20}
    net.addLink(s23, h5, cls=TCLink , **s23h5)

    info( '*** Starting network\n')
    net.build()
    info( '*** Starting controllers\n')
    for controller in net.controllers:
        controller.start()

    info( '*** Starting switches\n')
    net.get('s21').start([c0])
    net.get('s5').start([c0])
    net.get('s13').start([c0])
    net.get('s19').start([c0])
    net.get('s16').start([c0])
    net.get('s12').start([c0])
    net.get('s7').start([c0])
    net.get('s15').start([c0])
    net.get('s20').start([c0])
    net.get('s23').start([c0])
    net.get('s10').start([c0])
    net.get('s3').start([c0])
    net.get('s1').start([c0])
    net.get('s6').start([c0])
    net.get('s9').start([c0])
    net.get('s2').start([c0])
    net.get('s11').start([c0])
    net.get('s22').start([c0])
    net.get('s8').start([c0])
    net.get('s14').start([c0])
    net.get('s4').start([c0])
    net.get('s17').start([c0])
    net.get('s18').start([c0])

    info( '*** Post configure switches and hosts\n')

    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel( 'info' )
    myNetwork()

