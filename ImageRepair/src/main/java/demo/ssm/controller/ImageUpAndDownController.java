package demo.ssm.controller;

import demo.ssm.service.ImageUpAndDownService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.multipart.MultipartFile;

import javax.servlet.ServletContext;
import javax.servlet.http.HttpSession;
import java.io.File;
import java.io.IOException;
import java.util.UUID;

@Controller
public class ImageUpAndDownController {

    @Autowired
    private ImageUpAndDownService imageUpAndDownService;

    @RequestMapping(value = "/image/up",method = RequestMethod.POST)
    public String testUp(MultipartFile photo, HttpSession session, Model model) throws IOException {

        //获取上传的文件的文件名
        String fileName = photo.getOriginalFilename();
        //获取上传的文件的后缀名
        String hzName = fileName.substring(fileName.lastIndexOf("."));
        //获取uuid
        String uuid = UUID.randomUUID().toString();
        //拼接一个新的文件名
        fileName = uuid + hzName;
        //获取ServletContext对象
        ServletContext servletContext = session.getServletContext();
        //获取当前工程下photo目录的真实路径
        String photoPath = "D:\\code\\Java_web\\ImageRepair\\src\\main\\webapp\\image";
        String finalPath = photoPath + File.separator + fileName;
        //上传文件
        photo.transferTo(new File(finalPath));
        String dir = imageUpAndDownService.remoteCall(finalPath);
        model.addAttribute("u","/image/"+dir);
        return "index";
    }
}
