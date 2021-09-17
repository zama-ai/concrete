# Docker

## Setting up docker and X forwarding

Before you start this section, go ahead and install docker. You can follow [this](https://docs.docker.com/engine/install/) official guide for that.

### Linux

```shell
xhost +localhost
```

### Mac OS

To be able to use X forwarding on Mac OS:
- Install XQuartz
- Open XQuartz.app application, make sure in the application parameters that `authorize network connections` are set (currently in the Security settings)
- Open a new terminal within XQuartz.app and type:

```shell
xhost +127.0.0.1
```

and now, the X server should be all set in docker (in the regular terminal).

### Windows

Install Xming and use Xlaunch:
- Multiple Windows, Display number: 0
- `Start no client`
- **IMPORTANT**: Check `No Access Control`
- You can save this configuration to re-launch easily, then click finish.

## Logging in and building the image

Docker image of **Concrete** is based on another docker image provided by the compiler team. Once you have access to this repository you should be able to launch the commands to build the dev docker image with `make docker_build`.

Upon joining to the team, you need to log in using the following command:

```shell
docker login ghcr.io
```

This command will ask for a username and a password. For username, just enter your GitHub username. For password, you should create a personal access token from [here](https://github.com/settings/tokens) selecting `read:packages` permission. Just paste the generated access token as your password, and you are good to go.

Once you do that you can get inside the docker environment using the following command:

```shell
make docker_build_and_start
```

After you finish your work, you can leave the docker by using the `exit` command or by pressing `CTRL + D`.
