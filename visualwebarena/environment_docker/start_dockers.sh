#! /bin/bash

# Check if IP address is provided as first argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <IP_ADDRESS>"
    echo "Example: $0 3.136.54.71"
    exit 1
fi

IP_ADDRESS=$1

### 4. For VWA:
docker start shopping
docker start forum
# docker start kiwix33
# cd classifieds_docker_compose
# cat ./docker-compose.yml
# vi classifieds_docker_compose/docker-compose.yml  # Set CLASSIFIEDS to your site url, and change the reset token if required
# docker compose up --build -d

### 4. For WebArena:
docker start gitlab
docker start shopping_admin
# cd /home/ubuntu/openstreetmap-website/
# docker compose start
sleep 120

### 5. For VWA:
# docker exec classifieds_db mysql -u root -ppassword osclass -e 'source docker-entrypoint-initdb.d/osclass_craigslist.sql'  # Populate DB with content

docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="http://${IP_ADDRESS}:7770" # no trailing slash
docker exec shopping mysql -u magentouser -pMyPassword magentodb -e  "UPDATE core_config_data SET value='http://${IP_ADDRESS}:7770/' WHERE path = 'web/secure/base_url';"
docker exec shopping /var/www/magento2/bin/magento cache:flush

# # Disable re-indexing of products
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule catalogrule_product
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule catalogrule_rule
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule catalogsearch_fulltext
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule catalog_category_product
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule customer_grid
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule design_config_grid
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule inventory
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule catalog_product_category
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule catalog_product_attribute
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule catalog_product_price
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule cataloginventory_stock

### For WebArena:
docker exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set --base-url="http://${IP_ADDRESS}:7780"
docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e "UPDATE core_config_data SET value='http://${IP_ADDRESS}:7780/' WHERE path = 'web/secure/base_url';"
docker exec shopping_admin /var/www/magento2/bin/magento cache:flush

docker exec gitlab sed -i "s|^external_url.*|external_url 'http://${IP_ADDRESS}:8023'|" /etc/gitlab/gitlab.rb
docker exec gitlab gitlab-ctl reconfigure
