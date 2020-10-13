This script downloads data from the [BSRN](https://bsrn.awi.de/). It only downloads data from the station (CAM) and date range (2013-2015) that is relevant for REFIT.

- Install [Node.js](https://nodejs.org/en/) (tested with 12.18.0)
- Install [Yarn](https://classic.yarnpkg.com/en/docs/install)
- Install dependencies
  - Run `yarn` in this directory
- Compile
  - Run `yarn build` in this directory
- [Log in to Pangaea](https://www.pangaea.de/user/login.php)
  - Credentials are the same as the BSRN FTP credentials
- Get the `PanLoginID` and `pansessid` cookie values
  - In Chrome, open developer tools (Ctrl+Shift+I)
  - Go to `Application` -> `Cookies` -> `pangaea.de`
  - Copy the the cookie values from the `value` column
- Run the download script
  - Run `node dist <PanLoginID> <pansessid>` with the appropriate cookie values

The data will be downloaded to `data/bsrn/camMMYY.csv` relative to the repo root, where `MM` is the month and `YY` is the year.
