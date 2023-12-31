# Backend

## Backend service:
This is a JavaScript code built using Node.js, so make sure you have Node.js installed on your system.\
This service is using MySQL as the database, so you also have to run MySQL on your system.

1. Clone the repository then open it using your code editor.
2. Supposedly you have done the steps from the [predict-api](https://github.com/C22-PS088/predict-api) repository, you can continue these steps. If not, check the [predict-api](https://github.com/C22-PS088/predict-api) repository and finish the steps there first (otherwise you can't complete these steps).
3. In the root directory of this project, make a new file named __.env__ to provide the configurations needed.
4. Provide these details in the __.env__ file:
```
# Fill "" with the url of the predict api ex: http://localhost:5000
API_PREDICT_HOST=""
# Fill "" with your database username ex: root
DB_USERNAME=""
# Fill "" with your database password
DB_PASSWORD=""
# Fill "" with your database host name ex: localhost
DB_HOSTNAME=""
# This is the database name, no need to change it
DB_NAME="db_salindungi"
# No need to change this
ACCESS_TOKEN_SECRET="asdhj81dasdjh21fghagsfc93rfgbajfjkghw48iot"
# No need to change this
REFRESH_TOKEN_SECRET="adh19hbhg81gh823reh1980grh19fgh1f1fhasfja"
# Fill "" with your Midtrans serverKey
MIDTRANS_SERVER_KEY=""
# Fill "" with your Midtrans clientKey
MIDTRANS_CLIENT_KEY=""
# Fill "" with the bucket name you created in the previous step
GCS_BUCKET=""
# Fill "" with the project_id value from the sa-lindungi-credentials.json file
GCLOUD_PROJECT=""
# Fill "" with the client_email value from the sa-lindungi-credentials.json file
GCLOUD_CLIENT_EMAIL=""
# Fill "" with the private_key value from the sa-lindungi-credentials.json file
GCLOUD_PRIVATE_KEY=""
# Add whitelisted IP address for CORS ex: http://localhost;http://localhost:3000 (Use semicolon to separate)
WHITELISTED=""
```
5. Open terminal in the project root directory, then run `npm install` to install the dependencies.
6. Run these commands to configure the database migrations:\
`npx sequelize-cli db:create`\
`npx sequelize-cli db:migrate`\
`npx sequelize-cli db:seed:all`
7. Run the app using the command: `node ./bin/www`.
8. The server will run in the localhost with the port 8080, open [http://localhost:8080](http://localhost:8080) to view it in your browser.
9. If it doesn't show any errors then you have successfully run the service.
10. The next step is to configure the frontend service, you can find it in the [frontend](https://github.com/C22-PS088/frontend) repository.

You can check the public API documentation that we used for the mobile app [here](https://documenter.getpostman.com/view/19992713/Uz5CKdFu).

