# DJI Matrice 30 PSDK bridge boundary

This directory defines the local gRPC boundary between the Python research
runtime and a future companion-computer process linked against DJI PSDK 3.9.2.
No DJI SDK source, credentials, or proprietary libraries belong in this
repository.

The bridge must own control-authority negotiation, fused-position conversion to
takeoff-anchored local NED, HMS monitoring, RC-loss behavior, command rate
limiting, geofence enforcement, and a watchdog that hovers or lands when the
Python client stops sending fresh commands.

Generate Python stubs into `integrations/dji_psdk/generated/` only in a local
development environment. Generated stubs are portable; DJI libraries are not.
Hardware activation remains blocked until payload mass, power, thermal, E-Port,
firmware, and regulatory checks are documented.
