<?php
require_once('db.php');

if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_POST['ottoman']) && isset($_POST['turkish'])) {
    // Veri kaydetme işlemi için POST isteği gönder
    $ottoman = $_POST['ottoman'];
    $turkish = $_POST['turkish'];
    
    // Veritabanı bağlantısı
    $servername = "localhost";
    $username = "root";
    $password = "";
    $dbname = "ottoman";

    $conn = new mysqli($servername, $username, $password, $dbname);

    if ($conn->connect_error) {
        die("Veritabanı bağlantısı başarısız: " . $conn->connect_error);
    }

    // SQL sorgusu ile veriyi ekleyin
    $sql = "INSERT INTO translate (ottoman, turkish) VALUES ('$ottoman', '$turkish')";
    if ($conn->query($sql) === TRUE) {
        echo "success"; // Başarılı ekleme işlemi
    } else {
        echo "error"; // Hata durumunda
    }

    $conn->close();
    exit();
}

if ($_GET['action'] == 'get_data') {
    // Veritabanı bağlantısı
    $servername = "localhost";
    $username = "root";
    $password = "";
    $dbname = "ottoman";
    $conn = new mysqli($servername, $username, $password, $dbname);

    // Bağlantı hatası kontrolü
    if ($conn->connect_error) {
        die("Veritabanı bağlantısı başarısız: " . $conn->connect_error);
    }

    // SQL sorgusu
    $sql = "SELECT ottoman, turkish FROM translate";
    $result = $conn->query($sql);
    $data = array();

    // Veri kümesini dolaşarak verileri JSON formatına dönüştürme
    if ($result->num_rows > 0) {
        while ($row = $result->fetch_assoc()) {
            $ottoman_text = $row["ottoman"];
            $turkish_translation = $row["turkish"];
            $entry = array(
                "ottoman_text" => $ottoman_text,
                "turkish_translation" => $turkish_translation
            );
            $data[] = $entry;
        }
        $total_records = $result->num_rows;
        echo json_encode(array("total_records" => $total_records, "data" => $data));
    } else {
        echo json_encode(array("total_records" => 0, "data" => array()));
    }

    // Veritabanı bağlantısını kapat
    $conn->close();
    exit();
}


echo "Invalid request"; // Geçersiz bir istek durumunda yanıt verin
?>
