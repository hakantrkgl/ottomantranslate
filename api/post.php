<?php
if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_POST['ottoman']) && isset($_POST['turkish'])) {
    // Veritabanı bağlantısı
    $servername = "localhost";
    $username = "root";
    $password = "";
    $dbname = "ottoman";

    $conn = new mysqli($servername, $username, $password, $dbname);

    if ($conn->connect_error) {
        die("Veritabanı bağlantısı başarısız: " . $conn->connect_error);
    }

    // POST verilerini al
    $ottoman = $_POST['ottoman'];
    $turkish = $_POST['turkish'];

    // Veri boş değilse SQL sorgusu ile veriyi ekleyin
    if (!empty($ottoman) && !empty($turkish)) {
        $sql = "INSERT INTO translate (ottoman, turkish) VALUES ('$ottoman', '$turkish')";

        if ($conn->query($sql) === TRUE) {
            // Başarılı ekleme işlemi
            // AJAX yanıtı olarak success mesajı döndür
            echo "success";
        } else {
            // Hata durumunda AJAX yanıtı olarak error mesajı döndür
            echo "error";
        }

        // Veritabanı bağlantısını kapat
        $conn->close();
    } else {
        // Boş veri durumunda AJAX yanıtı olarak empty mesajı döndür
        echo "empty";
    }

    exit();
}

// Get Data butonuna tıklandığında JSON veriyi oluşturup indirme işlemi
if(isset($_GET['action']) && $_GET['action'] == 'get_data') {
    // Veritabanı bağlantısı
	
	    $servername = "localhost";
    $username = "root";
    $password = "";
    $dbname = "ottoman";
    $conn = new mysqli($servername, $username, $password, $dbname);

    if ($conn->connect_error) {
        die("Veritabanı bağlantısı başarısız: " . $conn->connect_error);
    }

    // SQL sorgusu ile verileri sorgula
    $sql = "SELECT ottoman, turkish FROM translate";
    $result = $conn->query($sql);

    if ($result->num_rows > 0) {
        $data = array();

        while($row = $result->fetch_assoc()) {
            $ottoman_text = $row["ottoman"];
            $turkish_translation = $row["turkish"];

            $entry = array(
                "ottoman_text" => $ottoman_text,
                "turkish_translation" => $turkish_translation
            );

            $data[] = $entry;
        }

        // JSON formatına çevir
        $json_data = json_encode(array("data" => $data), JSON_UNESCAPED_UNICODE);

        // Veritabanı bağlantısını kapat
        $conn->close();

        // JSON veriyi bir dosyaya yaz
        $file_name = "data.json";
        file_put_contents($file_name, $json_data);

        // Dosyayı indirme bağlantısını oluştur
        header("Content-Type: application/json");
        header("Content-Disposition: attachment; filename=$file_name");
        readfile($file_name);

        // Dosyayı sil
        unlink($file_name);

        exit();
    } else {
        // Hiç veri yoksa veya hata oluştuysa boş bir JSON döndür
        echo json_encode(array("data" => array()), JSON_UNESCAPED_UNICODE);
        exit();
    }
}
?>
