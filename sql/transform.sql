SELECT o.order_id, o.order_date, o.product_id, o.units, o.unit_price,
       p.category, o.units * o.unit_price AS revenue
FROM orders AS o JOIN products AS p ON o.product_id = p.product_id
ORDER BY o.order_date, o.order_id;
